import { pipeline, ImageSegmentationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "image-segmentation";

export default () => {
  describe("Image Segmentation", () => {
    describe("Panoptic Segmentation", () => {
      const model_id = "Xenova/detr-resnet-50-panoptic";
      /** @type {ImageSegmentationPipeline} */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of ImageSegmentationPipeline", () => {
        expect(pipe).toBeInstanceOf(ImageSegmentationPipeline);
      });

      it(
        "single",
        async () => {
          const image = await load_cached_image("cats");

          const output = await pipe(image);

          // First, check mask shapes
          for (const item of output) {
            expect(item.mask.width).toEqual(image.width);
            expect(item.mask.height).toEqual(image.height);
            expect(item.mask.channels).toEqual(1);
            delete item.mask; // No longer needed
          }

          // Next, compare scores and labels
          const target = [
            {
              score: 0.9920125603675842,
              label: "cat",
            },
            {
              score: 0.9987540245056152,
              label: "remote",
            },
            {
              score: 0.9994567036628723,
              label: "remote",
            },
            {
              score: 0.9733413457870483,
              label: "couch",
            },
            {
              score: 0.9993967413902283,
              label: "cat",
            },
          ];

          expect(output).toBeCloseToNested(target, 2);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("Semantic Segmentation", () => {
      const model_id = "Xenova/segformer_b0_clothes";
      /** @type {ImageSegmentationPipeline } */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "single",
        async () => {
          const image = await load_cached_image("man_on_car");

          const output = await pipe(image);

          // First, check mask shapes
          for (const item of output) {
            expect(item.mask.width).toEqual(image.width);
            expect(item.mask.height).toEqual(image.height);
            expect(item.mask.channels).toEqual(1);
            delete item.mask; // No longer needed
          }

          // Next, compare scores and labels
          const target = [
            { score: null, label: "Background" },
            { score: null, label: "Hair" },
            { score: null, label: "Upper-clothes" },
            { score: null, label: "Pants" },
            { score: null, label: "Left-shoe" },
            { score: null, label: "Right-shoe" },
            { score: null, label: "Face" },
            { score: null, label: "Right-leg" },
            { score: null, label: "Left-arm" },
            { score: null, label: "Right-arm" },
            { score: null, label: "Bag" },
          ];

          expect(output).toBeCloseToNested(target, 2);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });
};
