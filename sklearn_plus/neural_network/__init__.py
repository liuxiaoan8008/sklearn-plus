from __future__ import absolute_import

from .text_classification.models import *
from .named_entity_recognition.models import *

def deserialize(class_name):
    return globals()[class_name]
