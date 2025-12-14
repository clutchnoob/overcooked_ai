"""
Feature name mapping for Overcooked featurized states.

This mirrors the ordering produced by `OvercookedGridworld.featurize_state`
for two players and two closest pots. The resulting observation vector has
length 96, arranged per focal player as:

    [player_i_features (46), other_player_features (46), rel_pos (2), abs_pos (2)]

where player_i_features are built in the same insertion order as the
featurization code:
1) orientation one-hot (N, S, E, W)
2) held object one-hot (onion, soup, dish, tomato)
3) closest object deltas
   - onion (dx, dy)
   - tomato (dx, dy)
   - dish (dx, dy)
   - soup (dx, dy, n_onions, n_tomatoes)
   - serving (dx, dy)
   - empty_counter (dx, dy)
4) per-pot features for the two closest pots:
   exists, is_empty, is_full, is_cooking, is_ready,
   num_onions, num_tomatoes, cook_time, dx, dy
5) wall indicators for adjacent tiles (4 directions)

We provide:
- FEATURE_NAMES: ordered list of length 96
- FEATURE_INDEX_TO_NAME: index -> name mapping
- FEATURE_GROUPS: semantic groupings for plotting
"""

from __future__ import annotations

from typing import Dict, List

PLAYER_ORIENTATIONS = ["N", "S", "E", "W"]
HELD_OBJECTS = ["onion", "soup", "dish", "tomato"]
CLOSEST_OBJECTS = [
    ("onion", ["dx", "dy"]),
    ("tomato", ["dx", "dy"]),
    ("dish", ["dx", "dy"]),
    ("soup", ["dx", "dy", "n_onions", "n_tomatoes"]),
    ("serving", ["dx", "dy"]),
    ("empty_counter", ["dx", "dy"]),
]
POT_FIELDS = [
    "exists",
    "is_empty",
    "is_full",
    "is_cooking",
    "is_ready",
    "num_onions",
    "num_tomatoes",
    "cook_time",
    "dx",
    "dy",
]
WALL_DIRS = ["0", "1", "2", "3"]  # Direction indices used in featurization


def _player_feature_names(player_idx: int, num_pots: int = 2) -> List[str]:
    names: List[str] = []

    # 1) orientation
    names.extend([f"p{player_idx}_orientation_{ori}" for ori in PLAYER_ORIENTATIONS])

    # 2) held object one-hot (order matches IDX_TO_OBJ in featurizer)
    names.extend([f"p{player_idx}_holding_{obj}" for obj in HELD_OBJECTS])

    # 3) closest object features in featurizer order
    for obj_name, fields in CLOSEST_OBJECTS:
        for field in fields:
            names.append(f"p{player_idx}_closest_{obj_name}_{field}")

    # 4) pot features for each closest pot (pot_idx order)
    for pot_idx in range(num_pots):
        for field in POT_FIELDS:
            names.append(f"p{player_idx}_closest_pot_{pot_idx}_{field}")

    # 5) walls
    names.extend([f"p{player_idx}_wall_{d}" for d in WALL_DIRS])

    return names


def build_feature_names(num_players: int = 2, num_pots: int = 2) -> List[str]:
    """
    Returns a list of feature names matching featurization order for focal player 0.
    
    The observation for each player is player-centric, meaning player 0's view
    contains their own features first, then the other player's features, then
    relative and absolute positions. We only build names for the focal player's
    view (96 features for 2 players, 2 pots).
    """
    assert num_players == 2, "Current mapping assumes 2 players."
    feature_names: List[str] = []

    # Player 0's features come first in their own observation
    feature_names.extend(_player_feature_names(0, num_pots))
    
    # Then other player's features (player 1 in player 0's view)
    feature_names.extend(_player_feature_names(1, num_pots))
    
    # Relative position from player 0 to player 1
    feature_names.extend(["p0_relpos_to_p1_dx", "p0_relpos_to_p1_dy"])
    
    # Absolute position of player 0
    feature_names.extend(["p0_abspos_x", "p0_abspos_y"])

    if num_players == 2 and num_pots == 2:
        assert len(feature_names) == 96, f"Expected 96 features, got {len(feature_names)}"
    return feature_names


FEATURE_NAMES: List[str] = build_feature_names()
FEATURE_INDEX_TO_NAME: Dict[int, str] = {idx: name for idx, name in enumerate(FEATURE_NAMES)}


def get_feature_name(idx: int) -> str:
    """Get human-readable name for feature at index idx."""
    return FEATURE_INDEX_TO_NAME.get(idx, f"feature_{idx}")


def _register_groups() -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(FEATURE_NAMES):
        if "_orientation_" in name:
            groups.setdefault("orientation", []).append(idx)
        elif "_holding_" in name:
            groups.setdefault("holding", []).append(idx)
        elif "_closest_onion_" in name:
            groups.setdefault("closest_onion", []).append(idx)
        elif "_closest_tomato_" in name:
            groups.setdefault("closest_tomato", []).append(idx)
        elif "_closest_dish_" in name:
            groups.setdefault("closest_dish", []).append(idx)
        elif "_closest_soup_" in name:
            groups.setdefault("closest_soup", []).append(idx)
        elif "_closest_serving_" in name:
            groups.setdefault("closest_serving", []).append(idx)
        elif "_closest_empty_counter_" in name:
            groups.setdefault("closest_empty_counter", []).append(idx)
        elif "_closest_pot_" in name:
            groups.setdefault("pots", []).append(idx)
        elif "_wall_" in name:
            groups.setdefault("walls", []).append(idx)
        elif "_relpos_" in name:
            groups.setdefault("relative_pos", []).append(idx)
        elif "_abspos_" in name:
            groups.setdefault("absolute_pos", []).append(idx)
        else:
            groups.setdefault("other", []).append(idx)
    return groups


FEATURE_GROUPS: Dict[str, List[int]] = _register_groups()

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_INDEX_TO_NAME",
    "FEATURE_GROUPS",
    "build_feature_names",
    "get_feature_name",
]
