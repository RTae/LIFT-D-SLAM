from src.utils.helper import argParser, logging
from src.provider import visualizes, feature
import src.services as services

def main():
    # Utils
    args = argParser.make_parser().parse_args()
    logging.setup_logging()

    # Feature extraction provider
    fe = feature.LIFTFeature()
    # Visulizer provider
    vtdv = visualizes.ThreeDViewer()

    c = services.Camera(args)
    t = services.Tracking()
    m = services.Mapping(fe)
    v = services.Visulizer(vtdv)
    while c.is_open():
        # Get frame from camera
        frame = c.run()
        # Extract it
        frame_data = t.run(vtdv, frame, fe)
        # Mapping feature
        frame_f = m.run(frame, frame_data, vtdv)
        # Show the result
        v.show(frame, frame_f)

if __name__ == '__main__':
    main()