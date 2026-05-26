/* ── Scroll Progress ─────────────────────────────────────────── */
const bar = document.getElementById('scrollProgress');
window.addEventListener('scroll', () => {
  const pct = window.scrollY / (document.body.scrollHeight - window.innerHeight);
  bar.style.transform = `scaleX(${pct})`;
});

/* ── Theme Toggle ────────────────────────────────────────────── */
const btn = document.getElementById('themeToggle');
const icon = btn.querySelector('i');
const saved = localStorage.getItem('theme');
if (saved === 'dark') { document.body.classList.add('dark'); icon.className = 'ph-bold ph-sun'; }

btn.addEventListener('click', () => {
  const isDark = document.body.classList.toggle('dark');
  icon.className = isDark ? 'ph-bold ph-sun' : 'ph-bold ph-moon';
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
});

/* ── Chapter Nav Dots ────────────────────────────────────────── */
const chapters = document.querySelectorAll('[data-chapter]');
const dots = document.querySelectorAll('.nav-dot');

const chapterObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const idx = parseInt(e.target.dataset.chapter, 10) - 1;
      dots.forEach(d => d.classList.remove('active'));
      if (dots[idx]) dots[idx].classList.add('active');
    }
  });
}, { threshold: 0.4 });

chapters.forEach(c => chapterObserver.observe(c));

dots.forEach((dot, i) => {
  dot.addEventListener('click', () => {
    const target = document.querySelector(`[data-chapter="${i + 1}"]`);
    if (target) target.scrollIntoView({ behavior: 'smooth' });
  });
});

/* ── Fade In on Scroll ───────────────────────────────────────── */
const fadeEls = document.querySelectorAll('.fade-in');
const fadeObserver = new IntersectionObserver((entries) => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = `${i * 0.05}s`;
      e.target.classList.add('visible');
      fadeObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.15 });

fadeEls.forEach(el => fadeObserver.observe(el));

/* ── Feature Bar Animation ───────────────────────────────────── */
const featFills = document.querySelectorAll('.feat-fill');
const barObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('animated');
      barObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.3 });

featFills.forEach(f => barObserver.observe(f));

/* ── Dataset Tabs ────────────────────────────────────────────── */
const tabBtns  = document.querySelectorAll('.tab-btn');
const panels   = document.querySelectorAll('.sheet-panel');

function activateTab(tabId) {
  tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  panels.forEach(p => {
    const isTarget = p.id === `sheet-${tabId}`;
    if (isTarget) {
      p.classList.add('active');
      // trigger entrance if section already visible
      requestAnimationFrame(() => p.classList.add('visible'));
      p.classList.remove('hiding');
    } else {
      p.classList.remove('active', 'visible', 'hiding');
    }
  });
}

tabBtns.forEach(btn => {
  btn.addEventListener('click', () => activateTab(btn.dataset.tab));
});

/* ── Sheet Panel Scroll In / Out ─────────────────────────────── */
const sheetObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    const panel = e.target;
    if (!panel.classList.contains('active')) return;
    if (e.isIntersecting) {
      panel.classList.remove('hiding');
      panel.classList.add('visible');
    } else {
      panel.classList.remove('visible');
      panel.classList.add('hiding');
    }
  });
}, {
  threshold: 0.1,
  rootMargin: '0px 0px -80px 0px'
});

panels.forEach(p => sheetObserver.observe(p));
