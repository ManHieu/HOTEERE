from abc import ABC, abstractmethod
import pdb
import re
from typing import Dict, List, Set, Tuple


class BaseOutputFormat(ABC):
    name = None

    @abstractmethod
    def format_output(self, *args, **kwargs) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def find_trigger_position(self, *args, **kwargs) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        raise NotImplementedError


OUTPUT_FORMATS : Dict[str, BaseOutputFormat] = {}


def register_output_format(format_class: BaseOutputFormat):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class


@register_output_format
class ECIOuputFormat(BaseOutputFormat):
    name = 'ECI_output'
    exitting_rel_template = "Yes. [{head}] {label} [{tail}] and {detail}"
    non_rel_template = "No. [{head}] {label} [{tail}] and {detail}"

    def format_output(self, 
                    head: str, tail: str, label: str,
                    important_words: List[Tuple[int, str]] = None, 
                    num_before_head_subword: int = None, 
                    num_before_tail_subword: int = None) -> str:
        # sorted as appearance order
        if important_words != None:
            important_words.sort(key=lambda x: x[0])
            detail = []
            for item in important_words:
                if item[0] == num_before_head_subword or item[0] == num_before_tail_subword:
                    assert item[1] == head or item[1] == tail
                    detail.append(f'[{item[1]}]')
                else:
                    detail.append(item[1])
            detail = ' '.join(detail)
            if label != "has no relation to":
                template = self.exitting_rel_template.format(head=head, label=label, tail=tail, detail=detail)
            else:
                template = self.non_rel_template.format(head=head, label=label, tail=tail, detail=detail)
            return template
        else:
            if label != "has no relation to":
                template = "Yes. [{head}] {label} [{tail}]".format(head=head, label=label, tail=tail)
            else:
                template = "No. [{head}] {label} [{tail}]".format(head=head, label=label, tail=tail)
            return template
    
    def find_trigger_position(self, generated_seq: str, head: str, tail: str):
        head = f'[{head}]'
        tail = f'[{tail}]'
        re_head = re.compile(re.escape(head), flags=re.I)
        re_tail = re.compile(re.escape(tail), flags=re.I)
        try:
            head_position = [(m.start() + 1, m.start() + len(head) - 1) for m in re.finditer(re_head, generated_seq)]
            tail_position = [(m.start() + 1, m.start() + len(tail) - 1) for m in re.finditer(re_tail, generated_seq)]
        except:
            pdb.set_trace()
        return head_position, tail_position
        


