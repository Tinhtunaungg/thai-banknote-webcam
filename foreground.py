"""Refine an existing perspective crop; this stage does not locate banknotes.

Skin colour is only a probable-background hint, never a hard exclusion rule.
It is imperfect under different lighting and can overlap banknote ink.
"""

import cv2
import numpy as np


def refine_banknote(crop, enabled=True):
    """Return mask, preview, coverage and status without altering the crop.

    A narrow crop rim provides background seeds. Non-skin pixels in the
    interior provide foreground seeds. Border-connected skin candidates are
    probable background so GrabCut can revise the colour heuristic.
    """
    if crop is None:
        return {"mask": None, "preview": None, "coverage": 0.0,
                "status": "no crop", "usable": False}

    height, width = crop.shape[:2]
    full_mask = np.full((height, width), 255, dtype=np.uint8)
    if not enabled:
        return {"mask": full_mask, "preview": crop.copy(), "coverage": 1.0,
                "status": "disabled", "usable": True}

    def rejected(status):
        return {"mask": np.zeros_like(full_mask), "preview": np.zeros_like(crop),
                "coverage": 0.0, "status": status, "usable": False}

    if min(height, width) < 20:
        return rejected("crop too small")

    scale = min(1.0, 320.0 / max(height, width))
    small = cv2.resize(crop, (round(width * scale), round(height * scale)))
    h, w = small.shape[:2]
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    # Ignore isolated skin-coloured printed details: only regions connected
    # to the rim can seed a possible hand entering the crop.
    _, components = cv2.connectedComponents(skin)
    edge_ids = np.unique(np.concatenate((components[0], components[-1],
                                         components[:, 0], components[:, -1])))
    edge_ids = edge_ids[edge_ids != 0]
    possible_hand = np.isin(components, edge_ids)

    labels = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    labels[possible_hand] = cv2.GC_PR_BGD
    interior = np.zeros((h, w), dtype=np.uint8)
    interior[h // 5: h - h // 5, w // 5: w - w // 5] = 255
    interior[possible_hand] = 0
    interior = cv2.erode(interior, np.ones((3, 3), np.uint8))
    ambiguous_skin = cv2.countNonZero(interior) < 20
    if ambiguous_skin:
        # Pink/brown notes may fall almost entirely inside the skin range.
        # Discard that hint rather than declaring the whole note a hand.
        labels[:] = cv2.GC_PR_FGD
        interior[:] = 0
        interior[h // 5: h - h // 5, w // 5: w - w // 5] = 255
    if cv2.countNonZero(interior) < 20:
        return rejected("insufficient foreground seeds")
    labels[interior != 0] = cv2.GC_FGD
    labels[[0, -1], :] = cv2.GC_BGD
    labels[:, [0, -1]] = cv2.GC_BGD

    try:
        cv2.grabCut(small, labels, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 2, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return rejected("GrabCut could not refine crop")

    kept = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)
    mask = cv2.resize(kept, (width, height), interpolation=cv2.INTER_NEAREST)
    coverage = float(cv2.countNonZero(mask)) / mask.size
    usable = coverage >= 0.25
    return {"mask": mask, "preview": cv2.bitwise_and(crop, crop, mask=mask),
            "coverage": coverage, "usable": usable,
            "status": ("refined (ambiguous skin hints ignored)" if ambiguous_skin
                       else "refined") if usable else "too little foreground"}
