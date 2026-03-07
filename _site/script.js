(function() {
  // Dark mode — respect saved preference, then system preference
  var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  var saved = localStorage.getItem('theme');
  var isDark = saved === 'dark' || (!saved && prefersDark);
  if (isDark) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }

  // Dark mode toggle button
  var themeBtn = document.querySelector('.theme-toggle');
  if (themeBtn) {
    function updateIcon() {
      var dark = document.documentElement.getAttribute('data-theme') === 'dark';
      themeBtn.textContent = dark ? '\u2600' : '\u263E';
      themeBtn.setAttribute('title', dark ? 'Switch to light mode' : 'Switch to dark mode');
    }
    updateIcon();
    themeBtn.addEventListener('click', function() {
      var dark = document.documentElement.getAttribute('data-theme') === 'dark';
      if (dark) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('theme', 'light');
      } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
      }
      updateIcon();
    });
  }

  // Sidebar toggle
  var toggle = document.querySelector('.sidebar-toggle');
  var sidebar = document.querySelector('.sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', function() {
      sidebar.classList.toggle('open');
    });
  }

  // Active page highlighting
  var currentPath = window.location.pathname.replace(/\/index\.html$/, '/');
  document.querySelectorAll('.sidebar li[data-page]').forEach(function(li) {
    var link = li.querySelector('a');
    if (link && link.pathname && currentPath.endsWith(link.pathname.replace(/\/index\.html$/, '/'))) {
      li.classList.add('active');
    }
  });
})();
