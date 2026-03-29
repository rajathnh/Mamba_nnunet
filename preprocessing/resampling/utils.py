from typing import Callable

from batchgenerators.utilities.file_and_folder_operations import join
from utilities.find_class_by_name import recursive_find_python_class
from paths import pancreas_nnunet_code


def recursive_find_resampling_fn_by_name(resampling_fn: str) -> Callable:
    ret = recursive_find_python_class(join(pancreas_nnunet_code, "preprocessing", "resampling"), resampling_fn,
                                      'preprocessing.resampling')
    if ret is None:
        raise RuntimeError("Unable to find resampling function named '%s'. Please make sure this fn is located in the "
                           "preprocessing.resampling module." % resampling_fn)
    else:
        return ret
