import { ImageProcessor } from "../src/image_processors_utils.js";
import { YolosImageProcessor } from "../src/models/yolos/image_processing_yolos.js";
import { RawImage } from "../src/utils/image.js";

describe("ImageProcessor resize", () => {
  it.each([
    ["landscape", 640, 480, [1066, 800]],
    ["portrait", 480, 640, [800, 1066]],
    ["square", 224, 224, [800, 800]],
    ["already at the shortest edge", 1200, 800, [1200, 800]],
    ["at the longest-edge limit", 1333, 800, [1333, 800]],
  ])("uses the shortest-edge target (%s)", (_, width, height, expected) => {
    const size = { shortest_edge: 800, longest_edge: 1333 };
    const processor = new ImageProcessor({ size });
    const image = new RawImage(new Uint8ClampedArray(width * height * 3), width, height, 3);

    expect(processor.get_resize_output_image_size(image, size)).toEqual(expected);
  });

  it.each([
    ["both edge limits", { shortest_edge: 800, longest_edge: 1200 }, 1600, 400, [1200, 300]],
    ["shortest edge only", { shortest_edge: 800 }, 640, 480, [1066, 800]],
    ["longest edge only, downscaling", { longest_edge: 800 }, 1280, 960, [800, 600]],
    ["longest edge only, upscaling", { longest_edge: 800 }, 320, 240, [800, 600]],
    ["explicit dimensions", { width: 640, height: 640 }, 320, 240, [640, 640]],
  ])("honors %s", (_, size, width, height, expected) => {
    const processor = new ImageProcessor({ size });
    const image = new RawImage(new Uint8ClampedArray(width * height * 3), width, height, 3);

    expect(processor.get_resize_output_image_size(image, size)).toEqual(expected);
  });

  it("applies size divisibility after resizing to the shortest edge", () => {
    const size = { shortest_edge: 512, longest_edge: 864 };
    const processor = new ImageProcessor({ size, size_divisibility: 16 });
    const image = new RawImage(new Uint8ClampedArray(640 * 480 * 3), 640, 480, 3);

    expect(processor.get_resize_output_image_size(image, size)).toEqual([672, 512]);
  });
});

describe("YolosImageProcessor resize", () => {
  it.each([
    ["default patch alignment", {}, [672, 512]],
    ["explicit size_divisibility", { size_divisibility: 8 }, [680, 512]],
    ["legacy size_divisor", { size_divisor: 8 }, [680, 512]],
  ])("honors %s", (_, config, expected) => {
    const size = { shortest_edge: 512, longest_edge: 864 };
    const processor = new YolosImageProcessor({ size, ...config });
    const image = new RawImage(new Uint8ClampedArray(640 * 480 * 3), 640, 480, 3);

    expect(processor.get_resize_output_image_size(image, size)).toEqual(expected);
  });
});
