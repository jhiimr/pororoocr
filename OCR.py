from time import time
from queue import Queue
from threading import Thread

import torch
from easyocr import Reader
from pororo.PororoOcr import PororoOcr

from config import PARSER


torch.device('cpu')


def wrapper(func, args, queue):
    queue.put(func(args))


class OCR:
    def __init__(self):
        self.parser = PARSER
        self.easyocr = Reader(lang_list=['ko', 'en'])
        self.pororoocr = PororoOcr()

    def run_easyocr(self, image):
        result = self.easyocr.readtext(image, detail = 0)
        result_text = self.parser.join(result)
        return result_text


    def run_pororoocr(self, image):
        result = self.pororoocr.run_ocr(image)
        result_text = self.parser.join(result)
        return result_text

    def run_ocr(self, image):
        ocr_runners = [self.run_easyocr, self.run_pororoocr]

        # === Run ocr
        queues = []

        for ocr in ocr_runners:
            queue = Queue()
            Thread(target=wrapper, args=(ocr, image, queue)).start()
            queues.append(queue)

        ocr_results = [queue.get() for queue in queues]

        return ocr_results
