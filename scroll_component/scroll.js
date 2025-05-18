// scroll_component/scroll.js
const root = document.getElementById("root");
let lastScrollTop = 0;

window.addEventListener("wheel", function (e) {
  const scrollDelta = Math.sign(e.deltaY);
  if (window.scrollY !== lastScrollTop) {
    lastScrollTop = window.scrollY;
    const value = Math.min(Math.max(window.scrollY, 0), 10000);
    Streamlit.setComponentValue(value);
  }
});

Streamlit.setComponentReady();
Streamlit.setFrameHeight(window.innerHeight);