from src.utils.helper.general import argParser
from src.provider import visualizes, feature
import src.services as services

def main():
    args = argParser.make_parser().parse_args()

    # Feature extraction provider
    f_orb_f = feature.ORBFeature()
    # Visulizer provider
    vd = visualizes.Display()
    vtdv = visualizes.ThreeDViewer()

    c = services.Camera(args)
    t = services.Tracking()
    m = services.Mapping()
    v = services.Visulizer(vd, vtdv)
    while c.is_open():
        # Get frame from camera
        frame = c.run()
        # Extract it
        frame_data = t.run(vtdv, frame, f_orb_f)
        # Mapping feature
        frame_f = m.run(frame, frame_data, vtdv)
        # Show the result
        v.show(frame, frame_f)

if __name__ == '__main__':
    main()