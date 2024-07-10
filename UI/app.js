Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });

    dz.on("addedfile", function() {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);
        }
    });

    dz.on("complete", function(file) {
        let imageData = file.dataURL;

        var url = "http://127.0.0.1:5500/classify_image";

        $.post(url, {
            image_data: file.dataURL
        }, function(data, status) {
            if (!data || data.length == 0) {
                $("#resultSection").hide();
                $("#error").show();
                return;
            }
            let identifiedPerson = data[0].class.replace("_", " ");
            $("#error").hide();
            $("#resultSection").show();
            $("#resultName").text(identifiedPerson);
        });
    });

    $("#submitBtn").on('click', function(e) {
        dz.processQueue();
    });
}

$(document).ready(function() {
    $("#error").hide();
    $("#resultSection").hide();

    init();
});
