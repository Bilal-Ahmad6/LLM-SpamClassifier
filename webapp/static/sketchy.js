// Basic modal utility for popups and footer links
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.modal').forEach(function(modal) {
    var closeBtn = modal.querySelector('.close-modal, #closeModal');
    if (closeBtn) {
      closeBtn.onclick = function() {
        modal.style.display = 'none';
      };
    }
    modal.onclick = function(event) {
      if (event.target === modal) {
        modal.style.display = 'none';
      }
    };
  });

  // Footer modal links
  ['privacy-link', 'terms-link', 'help-link'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el && window.modalContents) {
      el.onclick = function(e) {
        e.preventDefault();
        var modal = document.getElementById('modal');
        var modalBody = document.getElementById('modal-body');
        if (modal && modalBody && window.modalContents[id]) {
          modalBody.innerHTML = window.modalContents[id];
          modal.style.display = 'block';
        }
      };
    }
  });
});
