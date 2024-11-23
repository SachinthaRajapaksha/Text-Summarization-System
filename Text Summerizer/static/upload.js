document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    const fileNameSpan = document.getElementById('file-name');
    const fileDisplay = document.getElementById('file-display');

    fileInput.addEventListener('change', function(e) {
        if (this.files && this.files.length > 0) {
            const fileName = this.files[0].name;
            fileNameSpan.textContent = 'File selected';
            fileDisplay.textContent = `Selected file: ${fileName}`;
            fileDisplay.style.display = 'block';
        } else {
            fileNameSpan.textContent = 'Choose a file';
            fileDisplay.textContent = '';
            fileDisplay.style.display = 'none';
        }
    });
});