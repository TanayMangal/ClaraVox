function toggleMode() {
  const on = document.body.classList.toggle("active");
  try { localStorage.setItem("claravox.contrast", on ? "1" : "0"); } catch (e) {   }
  document.dispatchEvent(new Event("themechange"));
}
