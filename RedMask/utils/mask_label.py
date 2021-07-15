def mask_label(class_int):
    return {
        0: 'good_mask',
        1: 'no_mask',
        2: 'no_nose',
        3: 'no_nose_mouth',
        4: 'no_chin',
    }.get(class_int, "no_status")
