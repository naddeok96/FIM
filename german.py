def numeric_difference_fisher_form(input, net , directions, eps=1e-3, parameter_generator = None, pred_approx = -1):
    '''
    Computes the approximate quadratic form of the infinitesimal Fisher information
    per Input change in direction of ``directions``.

    :param input: Input of net. Should have batch size 1
    :param net: A nn.Module object with log_softmax output
    :param eps: Used to approximate the directional derivative
    :param directions: The direction in which to take the directional derivative
    :param parameter_generator: The parameters of net for which to take the directional derivative.
    Should iterate through len(directions) elements.
    Default (None) is all parameters of net
    :param pred_approx: If None, all classes will be considered.
    If positive this will be interpreted as the prediction to reduce the model to two output nodes.
    If -1, argmax will be applied to the output of net and the output is reduced to two output nodes.
    :return: A torch tensor of the same dimension as input.
    '''

    # Compute output of model
    test_output = net(input)

    # Find number of classes model is considering
    output_dim = test_output.shape[1]

    # Check to make sure there is more than one class and standard pytorch output
    assert output_dim > 1
    assert test_output.shape[0] == 1 and len(test_output.shape) == 2

    # If we want to force the attack to perdict certain classes
    if pred_approx is not None:
        # Reduce the output to two dimensions
        assert pred_approx<output_dim
        if pred_approx == -1:
            _,pred = torch.max(test_output, dim=1)
        else:
            # to match shapes below
            pred = torch.tensor([pred_approx])
        def log_softmax_output(x):
            # j dummy variable
            full_linear_output = net(x)
            assert type(pred) is torch.Tensor
            pred_output = full_linear_output[:,pred]
            sum_without_pred = torch.logsumexp(full_linear_output[:, torch.arange(output_dim) != pred], dim=1, keepdim=True)
            reduced_linear_output = torch.cat((pred_output, sum_without_pred), 1)
            return reduced_linear_output

    # If all classes are to be considered 
    else:
        # Define the log softmax function
        log_softmax_output = lambda x: F.log_softmax(net(x), dim =1)

    # Define the softmax output function
    softmax_output = lambda x: F.softmax(log_softmax_output(x), dim = 1)

    # Declare directions as v 
    v = directions

    # 
    if parameter_generator is None:
        parameter_generator = net.parameters()

    # Initalize the FIM
    fisher_sum = 0
    x = deepcopy(input.data)
    x.requires_grad = True
    def get_x_grad():
        # return the gradient w.r.t x and null all gradients
        assert x.grad is not None
        grad = deepcopy(x.grad.data)
        x.grad.zero_()
        net.zero_grad()
        return grad
    def update_parameters(epsilon):
        for i, par in enumerate(net.parameters()):
            par.data += epsilon * v[i]
    for j in range(output_dim):
        if pred_approx is not None:
            if j not in [0,1]:
                continue
        # + eps
        update_parameters(eps)
        log_softmax_output(x)[0, j].backward()
        new_plus_linear_grad = get_x_grad()
        softmax_output(x)[0, j].backward()
        new_plus_softmax_grad = get_x_grad()
        # - eps
        update_parameters(-2 * eps)
        log_softmax_output(x)[0, j].backward()
        new_minus_linear_grad = get_x_grad()
        softmax_output(x)[0, j].backward()
        new_minus_softmax_grad = get_x_grad()
        # reset and evaluate
        update_parameters(eps)
        fisher_sum += 1/(2*eps)**2 * ((new_plus_linear_grad-new_minus_linear_grad)
                                      * (new_plus_softmax_grad - new_minus_softmax_grad))

    return fisher_sum