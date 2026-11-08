import { pipeline, ObjectDetectionPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "object-detection";

export default () => {
  describe("Object Detection", () => {
    describe("yolos", () => {
      const model_id = "Xenova/yolos-tiny";
      /** @type {ObjectDetectionPipeline} */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of ObjectDetectionPipeline", () => {
        expect(pipe).toBeInstanceOf(ObjectDetectionPipeline);
      });

      it(
        "single + threshold",
        async () => {
          const image = await load_cached_image("cats");
          const output = await pipe(image, { threshold: 0.9 });

          const target = [
            {
              score: 0.9899669289588928,
              label: "remote",
              box: { xmin: 46, ymin: 72, xmax: 179, ymax: 119 },
            },
            {
              score: 0.9431672692298889,
              label: "remote",
              box: { xmin: 336, ymin: 79, xmax: 368, ymax: 192 },
            },
            {
              score: 0.9497547149658203,
              label: "cat",
              box: { xmin: 338, ymin: 19, xmax: 638, ymax: 372 },
            },
            {
              score: 0.9794249534606934,
              label: "cat",
              box: { xmin: 10, ymin: 53, xmax: 313, ymax: 470 },
            },
            {
              score: 0.9760302305221558,
              label: "remote",
              box: { xmin: 41, ymin: 72, xmax: 178, ymax: 119 },
            },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("tiny-random", () => {
      const model_id = "hf-internal-testing/tiny-random-DetrForObjectDetection";

      /** @type {ObjectDetectionPipeline} */
      let pipe;
      let images;

      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
        images = await Promise.all([load_cached_image("white_image"), load_cached_image("blue_image")]);
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of ObjectDetectionPipeline", () => {
        expect(pipe).toBeInstanceOf(ObjectDetectionPipeline);
      });

      describe("batch_size=1", () => {
        it(
          "default (threshold unset)",
          async () => {
            const output = await pipe(images[0]);
            const target = [];
            expect(output).toBeCloseToNested(target, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "default (threshold=0)",
          async () => {
            const output = await pipe(images[0], { threshold: 0 });
            const target = [
              { score: 0.020384468138217926, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020384423434734344, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020384378731250763, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.02038446068763733, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020384378731250763, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020384445786476135, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020384419709444046, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.02038433775305748, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.02038438245654106, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.02038438431918621, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.02038429118692875, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020384421572089195, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            ];
            expect(output).toBeCloseToNested(target, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      // TODO: Add batched support to object detection pipeline
      // describe('batch_size>1', () => {
      //     it('default (threshold unset)', async () => {
      //         const output = await pipe(images);
      //         console.log(output);
      //         const target = [];
      //         expect(output).toBeCloseToNested(target, 5);
      //     }, MAX_TEST_EXECUTION_TIME);
      //     it('default (threshold=0)', async () => {
      //         const output = await pipe(images, { threshold: 0 });
      //         console.log(output);
      //         const target = [];
      //         expect(output).toBeCloseToNested(target, 5);
      //     }, MAX_TEST_EXECUTION_TIME);
      // });

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });
};
