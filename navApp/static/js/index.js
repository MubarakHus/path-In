        var section1 = document.getElementById("section1");
        var section2 = document.getElementById("section2");
        var image = document.getElementById("image");

        window.addEventListener("scroll", function() {
            var scrollPercent = window.scrollY / (document.documentElement.scrollHeight - window.innerHeight);
            var translateYValue = -(scrollPercent * (section2.offsetHeight - window.innerHeight));

            section1.style.transform = "translateY(" + translateYValue + "px)";
        });

        var isDragging = false;
        var startingX;

        image.addEventListener("mousedown", function(event) {
            isDragging = true;
            startingX = event.clientX - image.offsetLeft;
            image.style.cursor = "grabbing";
        });

        image.addEventListener("mousemove", function(event) {
            if (isDragging) {
                var moveX = event.clientX - startingX;
                image.style.transform = "translateX(" + moveX + "px)";
            }
        });

        image.addEventListener("mouseup", function() {
            isDragging = false;
            image.style.cursor = "grab";
        });

        image.addEventListener("mouseleave", function() {
            isDragging = false;
            image.style.cursor = "grab";
        });
        function showFloorMap(floorNumber) {
    // Hide all floor maps
    var maps = document.querySelectorAll('.floor-map');
    maps.forEach(function(map) {
        map.style.display = 'none';
    });

    // Show the selected floor map
    var selectedMap = document.getElementById('floor' + floorNumber);
    if (selectedMap) {
        selectedMap.style.display = 'block';
    }
}

// Call showFloorMap for the 1st floor on page load
document.addEventListener('DOMContentLoaded', function() {
    showFloorMap(1); // Adjust the index if your floor numbering is different
});