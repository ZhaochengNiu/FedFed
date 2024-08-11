import torch
# 导入 PyTorch 库，PyTorch 是一个开源的机器学习库，广泛用于深度学习。

# 这段代码是一个用于创建优化器的函数，它根据提供的参数和上下文（如角色和算法类型）来选择合适的优化器。以下是对代码的逐行解释：
# 整体来看，这个函数提供了一个灵活的方式来创建优化器，支持不同的角色和算法配置。对于服务器和客户端，
# 它可以根据提供的参数和上下文来选择创建 SGD 或 Adam 优化器，或者抛出异常。


def create_optimizer(args, model=None, params=None, **kwargs):
    # 定义了一个名为 create_optimizer 的函数，它接受以下参数：
    # args：一个包含优化器配置的参数对象。
    # model：一个 PyTorch 模型，其参数将被用于优化。
    # params：一个可选的参数列表，如果提供，则用于优化而不是 model 的参数。
    # **kwargs：额外的关键字参数，可以覆盖 args 中的设置。
    if "role" in kwargs:
        role = kwargs["role"]
    else:
        role = args.role
    # 确定优化器是为“服务器”还是“客户端”创建的。优先使用 kwargs 中的 role，如果没有则回退到 args.role。

    # params has higher priority than model
    if params is not None:
        params_to_optimizer = params
        # 如果提供了 params，则使用这些参数进行优化。
    else:
        if model is not None:
            params_to_optimizer = model.parameters()
        else:
            raise NotImplementedError
        pass
        # 如果没有提供 params，则使用 model 的参数。如果没有提供 model 或 params，则抛出 NotImplementedError 异常。
    if (role == 'server') and (args.algorithm in ['FedAvg']):
        # 如果角色是服务器并且算法是联邦平均（FedAvg），则根据服务器端的优化器设置创建优化器。
        if args.server_optimizer == "sgd":
            # 如果服务器端优化器设置为 SGD，则创建一个 SGD 优化器实例。
            # optimizer = torch.optim.SGD(params_to_optimizer,
            #     lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params_to_optimizer),
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
            # 使用 filter 函数筛选出需要梯度的参数，并创建 SGD 优化器实例。
        elif args.server_optimizer == "adam":
            # 如果服务器端优化器设置为 Adam，则创建一个 Adam 优化器实例。
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params_to_optimizer),
                lr=args.lr, weight_decay=args.wd, amsgrad=True)
            # 使用筛选出的参数创建 Adam 优化器实例，并设置 amsgrad=True 以使用 AMSGrad 变体。
        elif args.server_optimizer == "no":
            # 如果服务器端没有指定优化器，则创建一个默认的 SGD 优化器。
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params_to_optimizer),
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        else:
            raise NotImplementedError
            # 如果服务器端指定了不识别的优化器类型，则抛出 NotImplementedError 异常。
    else:
        # 如果角色不是服务器（可能是客户端），则根据客户端的优化器设置创建优化器。
        if args.client_optimizer == "sgd":
            # 如果客户端优化器设置为 SGD，则创建一个 SGD 优化器实例。
            optimizer = torch.optim.SGD(params_to_optimizer,
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        elif args.client_optimizer == "adam":
            raise NotImplementedError
            # 如果客户端优化器设置为 Adam，当前实现会抛出 NotImplementedError 异常。
        elif args.client_optimizer == "no":
            # 如果客户端没有指定优化器，则创建一个默认的 SGD 优化器。
            optimizer = torch.optim.SGD(params_to_optimizer,
                lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=args.nesterov)
        else:
            raise NotImplementedError
            # 如果客户端指定了不识别的优化器类型，则抛出 NotImplementedError 异常。
    return optimizer
    # 返回创建的优化器实例。







