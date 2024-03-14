METHOD_REGISTRY = {
    "baco": {
        "use_perms": True,
        "perm_kernel": "aug_spearman",
        "model_id": "gp_to",
        "use_tr": False,
    },
    "bops": {
        "use_perms": True,
        "perm_kernel": "mallows",
        "model_id": "gp_o",
        "use_tr": False,
    },
    "bodi_noperm": {
        "use_perms": False,
        "perm_kernel": "kendall",
        "model_id": "gp_hed",
        "use_tr": False,
    },
    "bodi_perm": {
        "use_perms": True,
        "perm_kernel": "kendall",
        "model_id": "gp_hed",
        "use_tr": False,
    },
    "casmo_noperm": {
        "use_perms": False,
        "perm_kernel": "kendall",
        "model_id": "gp_to",
        "use_tr": True,
    },
    "casmo_perm": {
        "use_perms": True,
        "perm_kernel": "kendall",
        "model_id": "gp_to",
        "use_tr": True,
    },
}