from typing import Type

from preprocessing.normalization.default_normalization_schemes import CTNormalization, NoNormalization, ImageNormalization

channel_name_to_normalization_mapping = {
    'ct': CTNormalization,
    'nonorm': NoNormalization,
}


def get_normalization_scheme(channel_name: str) -> Type[ImageNormalization]:
    """
    If we find the channel_name in channel_name_to_normalization_mapping return the corresponding normalization. If it is
    not found, use the default (CTNormalization)
    """
    norm_scheme = channel_name_to_normalization_mapping.get(channel_name.casefold())
    if norm_scheme is None:
        norm_scheme = CTNormalization
    # print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme
