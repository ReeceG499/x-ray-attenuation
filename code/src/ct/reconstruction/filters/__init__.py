from .ramlak import RamLakFilter
from .base import Filter

FILTERS = {
    "ramlak": RamLakFilter(),
}

def get_filter(name: str):
    return FILTERS[name.lower()]