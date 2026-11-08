import { RawImage, rand } from "../../src/transformers.js";
import { ImageProcessor } from "../../src/image_processors_utils.js";
import { load_cached_image } from "../asset_cache.js";

const TEST_IMAGES = {
  rgba: new RawImage(new Uint8ClampedArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), 2, 3, 4),
  rgb: new RawImage(new Uint8ClampedArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), 2, 3, 3),
  la: new RawImage(new Uint8ClampedArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), 2, 3, 2),
  l: new RawImage(new Uint8ClampedArray([0, 1, 2, 3, 4, 5]), 2, 3, 1),
};

describe("Image utilities", () => {
  describe("Resizing", () => {
    const uniform_cases = [
      ["upscale", 10, 10, 20, 20],
      ["mixed scales", 300, 200, 224, 224],
      ["single pixel", 1, 1, 7, 5],
    ];
    const colors = [[200], [23, 117, 209], [23, 117, 209, 255]];

    for (const resample of ["nearest", "bilinear", "bicubic", "lanczos"]) {
      for (const color of colors) {
        it.each(uniform_cases)(`${resample} preserves a uniform ${color.length}-channel image: %s`, async (_, width, height, new_width, new_height) => {
          const data = new Uint8ClampedArray(width * height * color.length);
          data.forEach((_, i) => {
            data[i] = color[i % color.length];
          });
          const original = data.slice();
          const image = new RawImage(data, width, height, color.length);
          const resized = await image.resize(new_width, new_height, { resample });

          expect(resized.size).toEqual([new_width, new_height]);
          expect(resized.channels).toBe(color.length);
          expect(resized.data.every((value, i) => value === color[i % color.length])).toBe(true);
          expect(image.data).toEqual(original);
          expect(image.size).toEqual([width, height]);
        });
      }
    }

    it.each(["nearest", "bilinear", "bicubic", "lanczos"])("preserves unsupported two-channel conversion behavior with %s", async (resample) => {
      const image = new RawImage(new Uint8ClampedArray([200, 255]), 1, 1, 2);
      await expect(image.resize(7, 5, { resample })).rejects.toThrow("Conversion failed due to unsupported number of channels");
    });

    // Reference values from sharp's native nearest, linear, and cubic resize kernels.
    it.each([
      [0, "nearest", [0, 0, 64, 64, 255, 255, 0, 0, 64, 64, 255, 255, 16, 16, 128, 128, 192, 192, 16, 16, 128, 128, 192, 192]],
      [2, "bilinear", [0, 0, 32, 64, 160, 255, 0, 0, 32, 64, 160, 255, 8, 8, 52, 96, 160, 224, 16, 16, 72, 128, 160, 192]],
      [3, "bicubic", [0, 0, 17, 60, 164, 255, 0, 0, 20, 64, 164, 255, 3, 8, 44, 96, 166, 224, 9, 16, 68, 128, 167, 192]],
    ])("uses the selected kernel for %s / %s", async (resample, name, expected) => {
      const image = new RawImage(new Uint8ClampedArray([0, 64, 255, 16, 128, 192]), 3, 2, 1);
      for (const method of [resample, name]) {
        const resized = await image.resize(6, 4, { resample: method });
        expect(resized.channels).toBe(1);
        expect(Array.from(resized.data)).toEqual(expected);
      }
    });

    it.each(["nearest", "bilinear", "bicubic"])("preserves uniform alpha with %s", async (resample) => {
      const color = [64, 128, 192, 128];
      const image = new RawImage(new Uint8ClampedArray(color), 1, 1, 4);
      const resized = await image.resize(7, 5, { resample });
      expect(resized.channels).toBe(4);
      expect(resized.data.every((value, i) => (i % 4 === 3 ? value === 128 : Math.abs(value - color[i % 4]) <= 1))).toBe(true);
    });

    it.each([
      [null, 2, [5, 2]],
      [2, null, [2, 1]],
      [2.5, 1.5, [3, 2]],
    ])("preserves rounded dimensions for %s x %s", async (width, height, expected) => {
      const image = new RawImage(new Uint8ClampedArray(7 * 3 * 3).fill(200), 7, 3, 3);
      expect((await image.resize(width, height)).size).toEqual(expected);
    });

    it("preserves no-op and unsupported resampling behavior", async () => {
      const image = new RawImage(new Uint8ClampedArray(3 * 2 * 3).fill(200), 3, 2, 3);
      expect(await image.resize(3, 2)).toBe(image);
      expect(await image.resize(null, null)).toBe(image);
      await expect(image.resize(6, 4, { resample: "unsupported" })).rejects.toThrow("Resampling method unsupported is not supported.");
    });

    it.each([2, 3])("keeps borders uniform after processor normalization with resample %s", async (resample) => {
      const image = new RawImage(new Uint8ClampedArray(10 * 10 * 3).fill(200), 10, 10, 3);
      const processor = new ImageProcessor({ size: { width: 20, height: 20 }, resample, do_normalize: true, image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5] });
      const { pixel_values } = await processor(image);
      const expected = (200 / 255 - 0.5) / 0.5;
      expect(pixel_values.dims).toEqual([1, 3, 20, 20]);
      expect(pixel_values.data.every((value) => Math.abs(value - expected) < 1e-6)).toBe(true);
    });
  });

  describe("Padding", () => {
    it("should pad image", async () => {
      /** @type {RawImage} */
      const padded_image = await load_cached_image("blue_image")
        .then((image) => image.resize(224, 224))
        .then((image) => image.pad([128, 128, 128, 128]));

      expect(padded_image.size).toEqual([480, 480]);

      const avg = padded_image.data.reduce((acc, val) => acc + val, 0) / padded_image.data.length;
      expect(avg).toBeCloseTo((224 * 224 * 255) / (3 * 480 * 480), 6);
    });
  });

  describe("Tensor to Image", () => {
    it("should create an image from a tensor (CHW)", () => {
      const tensor_chw = rand([3, 128, 256]).mul_(255).to("uint8");
      const image = RawImage.fromTensor(tensor_chw);
      expect(image.size).toEqual([256, 128]);
    });
    it("should create an image from a tensor (HWC)", () => {
      const tensor_hwc = rand([128, 256, 3]).mul_(255).to("uint8");
      const image = RawImage.fromTensor(tensor_hwc, "HWC");
      expect(image.size).toEqual([256, 128]);
    });
  });

  describe("Channel conversions", () => {
    it("should convert RGBA to L (grayscale)", async () => {
      const grayscale = TEST_IMAGES.rgba.clone().grayscale();
      expect(grayscale.size).toEqual(TEST_IMAGES.rgba.size);
      expect(grayscale.channels).toEqual(1);
    });

    it("should convert RGB to L (grayscale)", async () => {
      const grayscale = TEST_IMAGES.rgb.clone().grayscale();
      expect(grayscale.size).toEqual(TEST_IMAGES.rgb.size);
      expect(grayscale.channels).toEqual(1);
    });

    it("should convert L to RGB", async () => {
      const rgb = TEST_IMAGES.l.clone().rgb();
      expect(rgb.size).toEqual(TEST_IMAGES.l.size);
      expect(rgb.channels).toEqual(3);
    });

    it("should convert L to RGBA", async () => {
      const rgba = TEST_IMAGES.l.clone().rgba();
      expect(rgba.size).toEqual(TEST_IMAGES.l.size);
      expect(rgba.channels).toEqual(4);
    });

    it("should convert RGB to RGBA", async () => {
      const rgba = TEST_IMAGES.rgb.clone().rgba();
      expect(rgba.size).toEqual(TEST_IMAGES.rgb.size);
      expect(rgba.channels).toEqual(4);
    });

    it("should convert RGBA to RGB", async () => {
      const rgb = TEST_IMAGES.rgba.clone().rgb();
      expect(rgb.size).toEqual(TEST_IMAGES.rgba.size);
      expect(rgb.channels).toEqual(3);
    });
  });

  describe("putAlpha", () => {
    it("should add alpha to RGB image", async () => {
      const rgba = TEST_IMAGES.rgb.clone().putAlpha(TEST_IMAGES.l);
      expect(rgba.size).toEqual(TEST_IMAGES.rgb.size);
      expect(rgba.channels).toEqual(4);
    });
    it("should add alpha to RGBA image", async () => {
      const rgba = TEST_IMAGES.rgba.clone().putAlpha(TEST_IMAGES.l);
      expect(rgba.size).toEqual(TEST_IMAGES.rgb.size);
      expect(rgba.channels).toEqual(4);
    });
  });
});
