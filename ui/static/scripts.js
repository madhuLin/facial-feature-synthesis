document.getElementById('source_image').addEventListener('change', function() {
    var sourceImagesContainer = document.getElementById('source_images_preview');
    sourceImagesContainer.innerHTML = '';

    var files = this.files;
    for (var i = 0; i < files.length && i < 10; i++) {
        var file = files[i];
        var image = document.createElement('img');
        image.src = URL.createObjectURL(file);
        image.style.maxWidth = '100px';
        image.style.maxHeight = '100px';
        sourceImagesContainer.appendChild(image);
    }
});

document.getElementById('reference_image').addEventListener('change', function() {
    var referenceImagesContainer = document.getElementById('reference_images_preview');
    referenceImagesContainer.innerHTML = '';

    var files = this.files;
    for (var i = 0; i < files.length && i < 10; i++) {
        var file = files[i];
        var image = document.createElement('img');
        image.src = URL.createObjectURL(file);
        image.style.maxWidth = '100px';
        image.style.maxHeight = '100px';
        referenceImagesContainer.appendChild(image);
    }
});
