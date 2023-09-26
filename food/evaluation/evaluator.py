import time
import torch
import logging
import datetime
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.comm import is_main_process
from .calibration_layer import PrototypicalCalibrationBlock
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
import numpy,os,cv2
from detectron2.data.detection_utils import read_image

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator, cfg=None):
###111111111111111111111111111111111111111111111111111111111111111111111
    metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
    cpu_device = torch.device("cpu")
    instance_mode = ColorMode.IMAGE
    cfg.clone()
###111111111111111111111111111111111111111111111111111111111111111111111

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)

    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model),torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            ###111111111111111111111111111111111111111111111111111111111111111111111
            if cfg.TEST.SAVE_FIG:
                WINDOW_NAME = "VOC/COCO detections"
                for input in zip(inputs):
                    image = read_image(input[0]["file_name"], format="BGR")
                    image = image[:, :, ::-1]
                visualizer = Visualizer(
                    image, metadata, instance_mode=instance_mode
                )
                # if "instances" in outputs[0]:
                #     instances = outputs[0]["instances"].to(cpu_device)
                #     instances = instances[instances.scores > cfg.TEST.SCORE_THREHOLD]
                #     instances = [instances[instances.pred_classes == 15], instances[instances.pred_classes == 20]]
                #     for one_class_instance in instances:
                #         if one_class_instance[one_class_instance.pred_classes==20]:
                #             one_class_instance = one_class_instance[one_class_instance.scores > 0.44]
                #         vis_output = visualizer.draw_instance_predictions(
                #             predictions=one_class_instance
                #         )
                if "instances" in outputs[0]:
                    instances = outputs[0]["instances"].to(cpu_device)
                    instances = instances[instances.scores > cfg.TEST.SCORE_THREHOLD]
                    # instances = instances[instances.pred_classes == 20]
                    vis_output = visualizer.draw_instance_predictions(
                            predictions=instances
                        )
                if cfg.TEST.IMG_OUTPUT_PATH:

                    if not os.path.isdir(cfg.TEST.IMG_OUTPUT_PATH):
                        os.makedirs(cfg.TEST.IMG_OUTPUT_PATH)

                    if os.path.isdir(cfg.TEST.IMG_OUTPUT_PATH):
                        assert os.path.isdir(cfg.TEST.IMG_OUTPUT_PATH), cfg.TEST.IMG_OUTPUT_PATH
                        out_filename = os.path.join(
                            cfg.TEST.IMG_OUTPUT_PATH, os.path.basename(input[0]["file_name"])
                        )
                    else:
                        assert (
                                len(input) == 1
                        ), "Please specify a directory with args.output"
                        out_filename = cfg.TEST.IMG_OUTPUT_PATH
                    vis_output.save(out_filename)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(
                        WINDOW_NAME, vis_output.get_image()[:, :, ::-1]
                    )
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
            ###111111111111111111111111111111111111111111111111111111111111111111111

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
