import argparse
import StructureFromMotion as sfm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputDir', default='../Data/input', help='Input dir containing all the necessary files')
    parser.add_argument('--OutputDir', default='../Data/output', help='Output dir where files will be stored')
    parser.add_argument('--NumImages', default=6, help='Number of Images in the dir')

    args = parser.parse_args()
    input_dir = args.InputDir
    output_dir = args.OutputDir
    num_images = int(args.NumImages)

    sfm.run(input_dir, output_dir, num_images)


if __name__ == "__main__":
    main()
