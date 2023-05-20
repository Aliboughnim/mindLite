from django import template

register = template.Library()

@register.tag
def num_range(value):
    return range(value) 