from collections import defaultdict

from _0_RC_Model import Transient_Model


def main():
    model = Transient_Model()
    module_parameter_counts = defaultdict(int)

    print('Parameter details:')
    for name, parameter in model.named_parameters():
        parameter_count = parameter.numel()
        module_name = name.rsplit('.', 1)[0]
        module_parameter_counts[module_name] += parameter_count
        print(f'{name:<55} {str(tuple(parameter.shape)):<20} {parameter_count:>8,}')

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print('\nModule summary:')
    for module_name, parameter_count in module_parameter_counts.items():
        print(f'{module_name:<55} {parameter_count:>8,}')

    print(f'\nTotal parameters:     {total_parameters:,}')
    print(f'Trainable parameters: {trainable_parameters:,}')


if __name__ == '__main__':
    main()
