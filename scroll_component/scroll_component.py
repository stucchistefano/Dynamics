# scroll_component/scroll_component.py
import os
import streamlit as st
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "scroll_position",
    path=os.path.join(os.path.dirname(__file__), "")
)

def scroll_position():
    scroll_px = _component_func()
    return scroll_px or 0
