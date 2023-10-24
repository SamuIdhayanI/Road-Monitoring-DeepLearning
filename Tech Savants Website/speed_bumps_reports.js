const firebaseConfig = {
    apiKey: "AIzaSyDLAejGJ6EkNxHirZ-FddfJRaVH-N9w6v4",
    authDomain: "techsavants-bdf20.firebaseapp.com",
    projectId: "techsavants-bdf20",
    storageBucket: "techsavants-bdf20.appspot.com",
    messagingSenderId: "946186267157",
    appId: "1:946186267157:web:90019e46b74cd4bbbcfc1c",
    measurementId: "G-3CM8X2JZDN"
};
firebase.initializeApp(firebaseConfig);

let storage = firebase.storage();
let db = firebase.firestore();
var imagesRef = storage.ref('/image/');

imagesRef.listAll()
    .then((res) => {
        res.items.forEach((itemRef) => {
            itemRef.getDownloadURL().then((url) => {
                var imageContainer = document.createElement('div');
                imageContainer.className = "image-container";

                var img = document.createElement('img');
                img.src = url;
                img.className = "image-preview";

                imageContainer.appendChild(img);

                document.getElementById('imageDataContainer').appendChild(imageContainer);
            });
        });
    })
    .catch((error) => {
        console.log("Error:", error);
    });

db.collection("speed_bump")
    .get()
    .then((querySnapshot) => {
        if (!querySnapshot.empty) {
            querySnapshot.forEach((doc) => {
                const data = doc.data();
                createTable(data, doc.id); // Passing the document ID
            });
        } else {
            console.log("No documents found in 'speed_bump' collection.");
        }
    })
    .catch((error) => {
        console.error("Error fetching data:", error);
    });

function createTable(data, docId) {
    const tableContainer = document.getElementById("imageDataContainer");

    const table = document.createElement("table");
    table.className = "table table-bordered table-container mb-4 centered-table";
    table.style.width = "60%"; // Set the overall table width to 60%
    table.style.height = "200px"; // Increase the table height

    const row = table.insertRow();
    const imgCell = row.insertCell(0);
    const dataCell = row.insertCell(1);

    // Create an image element for the left side of the table (60% width)
    const img = document.createElement("img");
    const imageUrl = data.image_url;
    const imageName = imageUrl.split("/").pop();
    const firebaseStorageUrl = `https://firebasestorage.googleapis.com/v0/b/techsavants-bdf20.appspot.com/o/image%2Fspeed_bump%2F${encodeURIComponent(imageName)}?alt=media`;
    img.src = firebaseStorageUrl;
    img.alt = imageName;
    img.className = "image-preview";
    img.style.width = "100%"; // Set image width to 100%
    imgCell.appendChild(img);

    // Create Bootstrap buttons below the images
    const buttonsDiv = document.createElement("div");
    buttonsDiv.className = "mt-3"; // Add margin at the top
    const markAsRectifiedButton = document.createElement("button");
    markAsRectifiedButton.className = "btn btn-primary mr-2";
    markAsRectifiedButton.textContent = "Mark as Rectified";
    markAsRectifiedButton.addEventListener('click', function () {
        // Delete the document from Firestore
        db.collection("speed_bump").doc(docId).delete().then(function () {
            // Delete the image from storage
            storage.ref(`/image/speed_bump/${encodeURIComponent(imageName)}`).delete().then(function () {
                // Remove the table row from the HTML
                table.remove();
            }).catch(function (error) {
                console.error("Error deleting image from storage:", error);
            });
        }).catch(function (error) {
            console.error("Error deleting document from Firestore:", error);
        });
    });
    buttonsDiv.appendChild(markAsRectifiedButton);

    const showInMapButton = document.createElement("button");
    showInMapButton.className = "btn btn-info";
    showInMapButton.textContent = "Show in Map";
    showInMapButton.addEventListener('click', function () {
        viewOnMap(data['Lattitude'], data['Longitude']); // Call the function to open the map popup
    });
    buttonsDiv.appendChild(showInMapButton);
    imgCell.appendChild(buttonsDiv);

    // Create a data table for the right side of the table (40% width)
    const dataTable = document.createElement("table");
    dataTable.className = "table table-bordered";
    const dataOrder = ["Issue", "Date", "Time", "Lattitude", "Longitude", "Confidence"];
    for (const key of dataOrder) {
        if (key !== "timestamp") { // Exclude the timestamp
            const dataRow = dataTable.insertRow();
            const cell1 = dataRow.insertCell(0);
            const cell2 = dataRow.insertCell(1);
            cell1.innerHTML = key;
            cell2.innerHTML = data[key];
        }
    }
    dataCell.appendChild(dataTable);

    // Add the table to the container
    tableContainer.appendChild(table);

    table.addEventListener("mouseenter", function () {
        table.style.boxShadow = "0px 0px 15px rgba(0, 0, 0, 0.4)";
    });
    table.addEventListener("mouseleave", function () {
        table.style.boxShadow = "0px 0px 10px rgba(0, 0, 0, 0.2)";
    });
}

// Function to initialize the OpenStreetMap and display as a popup
function viewOnMap(lat, long) {
    // Create a Bootstrap modal element dynamically
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.tabIndex = -1;
    modal.role = 'dialog';
    modal.innerHTML = `
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <div id="map" style="height: 400px;"></div>
                </div>
            </div>
        </div>
    `;
    // Append the modal to the document body
    document.body.appendChild(modal);
    // Instantiate the modal using Bootstrap's function
    $(modal).modal('show');
    // When the modal is closed, remove it from the document body
    $(modal).on('hidden.bs.modal', function () {
        document.body.removeChild(modal);
    });

    // Initialize a map
    var map = L.map('map').setView([lat, long], 13);
    // Set up the OSM (Open Street Map) layer in the map
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    // Add a marker at the lat, long point
    L.marker([lat, long]).addTo(map);
}