from .parse import ParseClassificationDataset, ParseSegmentationDataset, ParseFromMapping
from .build import Build


class DataKit(object):
    ParseSegmentationDataset = ParseSegmentationDataset
    ParseClassificationDataset = ParseClassificationDataset
    ParseFromMapping = ParseFromMapping
    Build = Build
