"""Constants for ML preprocessing."""

import re
from enum import Enum
from typing import Final

from atomworks.enums import ChainType

PREPROCESSING_SUPPORTED_CHAIN_TYPES: Final[list[ChainType]] = [
    ChainType.NON_POLYMER,
    ChainType.POLYPEPTIDE_L,
    ChainType.POLYPEPTIDE_D,
    ChainType.DNA,
    ChainType.RNA,
    ChainType.BRANCHED,
    ChainType.DNA_RNA_HYBRID,
]
"""Chain types supported during preprocessing."""

PREPROCESSING_SUPPORTED_CHAIN_TYPES_INTS: Final[list[int]] = [t.value for t in PREPROCESSING_SUPPORTED_CHAIN_TYPES]
"""Integer values of chain types supported during preprocessing."""

TRAINING_SUPPORTED_CHAIN_TYPES: Final[list[ChainType]] = [
    ChainType.NON_POLYMER,
    ChainType.POLYPEPTIDE_L,
    ChainType.DNA,
    ChainType.RNA,
    ChainType.BRANCHED,
]
"""Chain types supported during training."""

TRAINING_SUPPORTED_CHAIN_TYPES_INTS: Final[list[int]] = [t.value for t in TRAINING_SUPPORTED_CHAIN_TYPES]
"""Integer values of chain types supported during training."""

PDB_REGEX: Final[re.Pattern[str]] = re.compile(r"^[0-9A-Za-z]{4}$")
"""Regex pattern for valid PDB IDs (4 alphanumeric characters)."""


class ClashSeverity(Enum):
    """Severity levels for atomic clashes in a structure."""

    SEVERE = "severe"
    """More than 50% of polymers are clashing."""

    MODERATE = "moderate"
    """Any polymers are clashing."""

    MILD = "mild"
    """Any clashes (polymer or non-polymer)."""

    NO_CLASH = "no-clash"
    """No clashes detected."""


class DistanceThresholds:
    """Centralized distance thresholds (Angstroms) for preprocessing."""

    CLASH: Final[float] = 1.0
    """Atoms closer than this are considered clashing."""

    CONTACT: Final[float] = 5.0
    """Distance threshold for atoms in contact."""

    INTERFACE: Final[float] = 15.0
    """Cutoff for interface atom identification in large assembly sampling."""


# fmt: off
PDB_IDS_THAT_CRASH_PREPROCESSING: Final[tuple[str, ...]] = (
    "1m4x", "1uf2", "3jan", "3jbp", "3k1q", "3zif", "4f5x", "4l3b", "4u3n", "4v5x", "5dat", "5j7v", "5jus", "5tbw", "5zap",
    "6b43", "6cgr", "6ftg", "6ftj", "6lgl", "6ncl", "6nhj", "6nu3", "6q1f", "6tb3", "6vyr", "6w19", "6woo", "6ylh", "6zvk",
    "7as5", "7bho", "7bsi", "7btb", "7bw6", "7fj1", "7lhd", "7nwh", "7qiz", "7r5j", "7r5k", "7tbi", "7tbk", "7v4t",
)
# fmt: on
"""PDB IDs that crash or hang the preprocessing pipeline.

Warning:
    These include extremely large assemblies (ribosomes, nuclear pore complexes)
    and complex structures (DNA origami) that exceed memory/time limits.
"""

# fmt: off
PDB_IDS_WITH_UNPHYSICAL_BONDS: Final[tuple[str, ...]] = (
    "185d", "193d", "1a0r", "1ay3", "1b4d", "1b6j", "1by5", "1c0t", "1c0u", "1c1b", "1can", "1cao", "1ci7", "1cpi", "1dtq", "1dtt", "1eva", "1evd", "1fav", "1fwm",
    "1gcm", "1jaa", "1jc8", "1jcp", "1jgc", "1l9w", "1niu", "1nma", "1nmb", "1pfe", "1pw8", "1qtn", "1qur", "1u8g", "1v4f", "1v6q", "1v7h", "1vs2", "1vtg", "1xhb",
    "1xql", "1xvk", "1xvn", "1xvr", "1zet", "2adw", "2bvr", "2bvs", "2c15", "2c3u", "2ddc", "2et0", "2fni", "2h77", "2j32", "2j58", "2jhg", "2llj", "2n4n", "2n6h",
    "2n6i", "2nbl", "2nmp", "2vlc", "2wdy", "2xe4", "2xjh", "2xji", "2xqt", "2yct", "3ah8", "3awo", "3bo7", "3bp2", "3f4z", "3go3", "3k87", "3kch", "3mgn", "3nuv",
    "3q9g", "3u4w", "3u51", "4auo", "4b3q", "4bp9", "4bqd", "4bxe", "4cak", "4ce5", "4chx", "4fa5", "4fa9", "4h0t", "4j5t", "4ks6", "4ktx", "4l1s", "4lnm", "4nec",
    "4oy5", "4rca", "4v15", "4w9f", "5aa2", "5bmm", "5bqc", "5c55", "5cs2", "5ctv", "5e83", "5eq7", "5eq8", "5f1w", "5g21", "5h1b", "5hkj", "5hwn", "5lgr", "5lvs",
    "5m3z", "5ol2", "5yty", "5ytz", "5zur", "6bpu", "6bpw", "6cdh", "6eik", "6euv", "6fbt", "6fcr", "6fcs", "6gmq", "6gmx", "6ibt", "6jze", "6m9z", "6skj", "6te5",
    "6tg4", "6th7", "6thq", "6uf4", "6uf7", "6uf8", "6uf9", "6vju", "6wve", "6x8k", "6x8l", "6xte", "6xu2", "6xu3", "6yfy", "6zuk", "7ajz", "7ak4", "7axp", "7axs",
    "7azd", "7bgq", "7dq0", "7dq8", "7f0c", "7f0f", "7jr4", "7oov", "7oow", "7opw", "7owm", "7pfz", "7pox", "7q1l", "7q9s", "7qgv", "7qtv", "7r0y", "7rnp", "7s50",
    "7ubz", "7v58", "7w3u", "7x6r", "7x97", "7x9f", "7xdj", "7z0l", "7z86", "7z9r", "7zss", "7zwh", "8a99", "8agy", "8ai7", "8an9", "8ao0", "8brn", "8cep", "8cx5",
    "8jug", "8sa8", "8sa9", "8sab", "8sng", "8snj", "8su6", "1b23", "1bcs", "1bjg", "1bwc", "1bx5", "1c2q", "1c2w", "1c4c", "1cfm", "1chw", "1d61", "1dd4", "1ddi",
    "1e1f", "1e3d", "1e7u", "1efd", "1f8c", "1f8d", "1fp4", "1fq5", "1fq8", "1g20", "1g21", "1g8j", "1ga5", "1gx8", "1h2j", "1hcy", "1hkx", "1hl9", "1i94", "1i95",
    "1i96", "1i97", "1j11", "1j12", "1j1a", "1jhi", "1jpc", "1kk4", "1kmk", "1ksj", "1l0l", "1m21", "1mbu", "1mbx", "1md2", "1n1h", "1n7d", "1nte", "1o7t", "1oao",
    "1pwf", "1q2c", "1q6k", "1q9x", "1qbv", "1qh1", "1qlb", "1qni", "1qnl", "1qnu", "1r6o", "1r6q", "1rpg", "1rqd", "1sgc", "1siw", "1sp4", "1sqb", "1sqq", "1sqv",
    "1sqx", "1t3p", "1th3", "1thc", "1tlf", "1twn", "1uwy", "1vje", "1vl3", "1vqw", "1w3l", "1w4r", "1w9m", "1wck", "1wdr", "1wvm", "1xv6", "1y4z", "1y6f", "1yfg",
    "1z7l", "2b7r", "2b7s", "2bjk", "2bvl", "2c3p", "2c5c", "2c6c", "2c6f", "2cch", "2cju", "2da8", "2dc7", "2dc8", "2dca", "2dcb", "2dcd", "2e1a", "2ef5", "2f1o",
    "2fwi", "2fyu", "2hhl", "2iwk", "2ja6", "2ja7", "2ja8", "2jji", "2m5b", "2min", "2nqx", "2o7a", "2olb", "2osm", "2p8x", "2rl7", "2uwn", "2uyn", "2uyv", "2uzs",
    "2v8f", "2v93", "2vzm", "2w31", "2wpu", "2x8j", "2xcr", "2xrz", "2xsj", "2y6e", "2yaj", "2yiv", "2ylj", "2ylo", "2zur", "3af7", "3bbf", "3bhv", "3c0v", "3c9e",
    "3cgb", "3d1l", "3d5x", "3dt0", "3en9", "3enh", "3esc", "3esd", "3f4i", "3h8l", "3hhm", "3hr3", "3i39", "3kku", "3kvs", "3l0o", "3lya", "3ni3", "3np2", "3ott",
    "3pa0", "3pot", "3pyk", "3q9h", "3q9i", "3q9j", "3rpd", "3rwr", "3rzs", "3sd3", "3sz0", "3szf", "3t0k", "3t2z", "3t4g", "3t5g", "3tuz", "3tvk", "3up9", "3usl",
    "3usm", "3uso", "3vah", "3w6a", "3ze9", "3zp9", "3zx0", "486d", "4a7n", "4a93", "4ag7", "4b09", "4b0n", "4c3a", "4c4m", "4car", "4cft", "4cpq", "4cps", "4cty",
    "4ctz", "4cu0", "4cu1", "4cul", "4cum", "4cun", "4cv5", "4cvg", "4cwv", "4cww", "4cwx", "4cwy", "4cwz", "4cx0", "4cx1", "4cx2", "4cyg", "4d34", "4d35", "4d36",
    "4d37", "4d38", "4d3a", "4daw", "4dlp", "4e0l", "4e0m", "4e0n", "4e0o", "4e54", "4e5z", "4ef1", "4ef2", "4eqw", "4eu9", "4fy0", "4g5f", "4gjs", "4got", "4i9b",
    "4ivh", "4j7v", "4j8w", "4j9u", "4jm0", "4js6", "4jsa", "4jsk", "4jsl", "4jsm", "4jss", "4k0k", "4k3f", "4k5h", "4k5i", "4k5j", "4k5k", "4kcp", "4kcq", "4kcr",
    "4kcs", "4kky", "4luw", "4mrr", "4n4k", "4n4l", "4n4m", "4n7k", "4n7l", "4n9r", "4nhp", "4nhq", "4nmm", "4ntp", "4ntr", "4nw8", "4nw9", "4o1w", "4ote", "4oxd",
    "4p4v", "4p4w", "4p4x", "4p4y", "4p4z", "4pa5", "4q5t", "4q8d", "4qgn", "4qo5", "4qp1", "4r6c", "4r7u", "4rlo", "4tt5", "4ttt", "4u6w", "4uaz", "4ubb", "4udk",
    "4uh7", "4uh8", "4uh9", "4uha", "4upq", "4upr", "4ups", "4upt", "4wc8", "4wna", "4x0s", "4xtd", "4xti", "4z12", "4z13", "4zc0", "4zjx", "4zqw", "5a4m", "5adj",
    "5adk", "5adl", "5adm", "5adn", "5brv", "5ci8", "5d51", "5dal", "5ddl", "5e9r", "5ecx", "5eul", "5f1t", "5fez", "5ff0", "5ff3", "5ff4", "5fj2", "5fj3", "5fle",
    "5fvy", "5fvz", "5giy", "5hes", "5hpp", "5ieg", "5ik9", "5iwd", "5jiw", "5kfp", "5lh6", "5lms", "5lpy", "5mb1", "5mdj", "5mdk", "5mdl", "5mf2", "5msg", "5mwx",
    "5n5r", "5ny6", "5nz6", "5oar", "5ob6", "5ob7", "5ob8", "5oeg", "5sur", "5sus", "5sut", "5suu", "5tpc", "5uhr", "5uod", "5v4g", "5v4h", "5v63", "5v64", "5v65",
    "5viv", "5vu9", "5vv7", "5vv8", "5vv9", "5vva", "5vvg", "5vvn", "5w4h", "5w4i", "5w4j", "5wnq", "5wnr", "5wns", "5wp6", "5x7h", "5zzf", "6b1r", "6b41", "6bo1",
    "6bo2", "6c6k", "6cdk", "6cg3", "6cg4", "6cg5", "6dr4", "6dr5", "6dr6", "6e1a", "6e5i", "6egw", "6ejw", "6erm", "6evj", "6f2f", "6f2h", "6f2i", "6f2j", "6f2m",
    "6frm", "6frn", "6gmi", "6gob", "6h8t", "6hf6", "6hf7", "6hkq", "6hmy", "6hsn", "6hso", "6hy6", "6i24", "6i25", "6ic8", "6jzm", "6kf5", "6ks1", "6lzb", "6m97",
    "6m9g", "6ma8", "6mik", "6mu5", "6nys", "6o7m", "6o9t", "6o9v", "6ofy", "6otl", "6pl9", "6pla", "6plb", "6pv2", "6q8j", "6qcv", "6qcw", "6qcx", "6qfu", "6qfv",
    "6qfx", "6qjs", "6qqi", "6qqk", "6qsa", "6qsh", "6qss", "6qxc", "6r2r", "6r93", "6rcf", "6rke", "6rqa", "6rr7", "6ruk", "6rve", "6s46", "6s4q", "6s50", "6s6w",
    "6s9z", "6swm", "6sym", "6tb2", "6tho", "6tvk", "6tvy", "6twc", "6u7y", "6udr", "6vfb", "6vjr", "6vu1", "6vu4", "6vvj", "6wgo", "6wnh", "6wxm", "6wxz", "6xuv",
    "6ycs", "6yh0", "6ykh", "6yo0", "6yxg", "6z4i", "6z7r", "6z8j", "6zre", "7a0d", "7a6q", "7abr", "7b0g", "7b3r", "7b97", "7bay", "7bpa", "7bpf", "7bpg", "7dix",
    "7efi", "7fbs", "7g96", "7jqr", "7jqs", "7jqt", "7jqu", "7jrh", "7jxn", "7jxo", "7jyy", "7jz0", "7ksv", "7ktm", "7kvv", "7l6r", "7l6t", "7lok", "7m4t", "7ne0",
    "7nh5", "7nwq", "7o1s", "7o25", "7o55", "7o6j", "7odg", "7odh", "7onm", "7onq", "7onv", "7otv", "7ph4", "7ph7", "7pr3", "7pwb", "7qdj", "7qhr", "7qpw", "7qpy",
    "7qpz", "7qy3", "7rnb", "7rsr", "7rtz", "7t4h", "7tne", "7tpn", "7tpo", "7tq0", "7tq9", "7tqc", "7tqe", "7tqf", "7tqh", "7tqi", "7tqj", "7tqk", "7tqw", "7u1b",
    "7uyn", "7uyo", "7uyp", "7vcn", "7vdt", "7wz9", "7xkj", "7y44", "7yd7", "7yd8", "7yeb", "7yec", "7yee", "7yx0", "7z60", "7z6a", "7z6d", "7zcy", "7zkn", "7zu8",
    "7zx5", "7zxl", "7zz3", "8a33", "8a5s", "8a6t", "8ab2", "8ai0", "8aqx", "8aqy", "8b7g", "8bqt", "8by0", "8cbz", "8crz", "8cs0", "8cs2", "8ec9", "8eca", "8es1",
    "8f4y", "8fo0", "8fxi", "8hyi", "8jeb", "8omy", "8on3", "8oqf", "8oqg", "8p4e", "8qm3", "8qml", "8w6p",
)
# fmt: on
"""PDB IDs with bond geometries that violate basic plausibility assumptions."""

PDB_IDS_TO_EXCLUDE: Final[frozenset[str]] = frozenset(PDB_IDS_THAT_CRASH_PREPROCESSING)
"""PDB IDs to exclude from preprocessing (only those that crash/hang)."""
