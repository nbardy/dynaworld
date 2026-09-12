"""Reproduce the measured F4 depth-order discontinuity without a GPU.

Values come from checkpoint ccf3d00c...e50c64a, frame 2, pixel (x=3,y=49),
primitives 178 and 242. This explains a failure; it is not a compiler fix.
Run from the repository root with Python (standard library only).
"""
import json
import struct


def float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def reproduce():
    time = 0.5
    depth0 = 2.180581569671631
    slopes = (-0.006274801678955555, 0.006274800281971693)
    anchors = (-0.48883867263793945, 1.4888386726379395)
    centered = [float32(depth0 + float32(b * float32(time - t)))
                for b, t in zip(slopes, anchors)]
    expanded = [float32(float32(depth0 - float32(b * t)) + float32(b * time))
                for b, t in zip(slopes, anchors)]
    # Adjacent alpha-compositing swap: C_ba - C_ab = T*a*b*(c_b-c_a).
    blue_delta = (0.7153020294779574 * 0.28963113135204804
                  * 0.28963113939013224
                  * (0.6417853832244873 - 0.6339662671089172))
    assert centered[0] == centered[1]
    assert expanded[0] > expanded[1]
    observed = 0.0004691779613494873
    assert abs(blue_delta - observed) < 1e-10
    return dict(
        centered_depth=centered, expanded_depth=expanded,
        centered_order=[178, 242], expanded_order=[242, 178],
        predicted_blue_change=blue_delta, observed_blue_change=observed,
        prediction_error=abs(blue_delta-observed),
        scope='measured float32 order-discontinuity reproduction, not a fix',
    )


if __name__ == '__main__':
    print(json.dumps(reproduce(), indent=2))
