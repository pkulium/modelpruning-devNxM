from typing import Any, Dict, Collection


def load_item_from_dict(
    lookup_location: Dict,
    lookup_key: str,
    default_value: Any,
    logger,
    validation_values: Collection = None,
) -> Any:

    ret_val = default_value

    try:
        ret_val = lookup_location[lookup_key]
    except KeyError:
        logger.error("JSON configuration error: Unable to find find {} in {}, using default".format(
            lookup_key,
            lookup_location
        ))
        ret_val = default_value

    if validation_values:
        if ret_val not in validation_values:
            logger.error("Configured value of {} is not valid for key {} (valid values: {}), using default".format(
                ret_val,
                lookup_key,
                validation_values
            ))
            ret_val = default_value

    return ret_val
