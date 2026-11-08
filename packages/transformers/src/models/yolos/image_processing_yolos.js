import { ImageProcessor, post_process_object_detection } from '../../image_processors_utils.js';

export class YolosImageProcessor extends ImageProcessor {
    constructor(config) {
        super(config);
        // Match YOLOS's patch alignment when it is omitted from the config.
        this.size_divisibility ??= 16;
    }

    /** @type {typeof post_process_object_detection} */
    post_process_object_detection(...args) {
        return post_process_object_detection(...args);
    }
}
export class YolosFeatureExtractor extends YolosImageProcessor {}
