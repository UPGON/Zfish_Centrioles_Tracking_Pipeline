import ezomero
import getpass
import os

HOST = "omero-server.epfl.ch"
PORT = 4064

class OMEROConnection:
    def __init__(self):
        super().__init__()
        self.conn = None
        # track whether the connection has been explicitly closed
        self._closed = True

    def __del__(self):
        # object is being garbage collected; ensure connection closed
        if not self._closed:
            try:
                self.disconnect()
            except Exception:
                pass

    # support context manager protocol so callers can use `with` blocks
    def __enter__(self):
        # establish connection lazily if not yet connected
        if self.conn is None or not getattr(self.conn, "isConnected", lambda: False)():
            self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def connect(self):
        username=str(input("Type Your Username:\n"))
        password=getpass.getpass()

        self.conn = ezomero.connect(user=username, password=password, group="",
                            host=HOST, port=PORT, secure=True)

        if self.conn is not None and self.conn.isConnected():
            print(f"Connected to {HOST}")
            self._closed = False
        else:
            print("ERROR: Not able to connect to OMERO server. Please check your credentials, group and hostname")

    def disconnect(self):
        if self.conn is not None and self.conn.isConnected():
            self.conn.close()
            self._closed = True
            print(f"Disconnect from {HOST}")
        else:
            # silently ignore repeated disconnects, but log for debugging
            print("(disconnect called but connection already closed)")


    def __show_projects__(self):
        print("List of available data in your session \n[Id: Name]")
        projects = self.conn.getObjects("Project", opts={'owner': self.conn.getUser().getId()})
        project_names = []
        for project in projects:
            print(str(project.getId())+" : "+str(project.getName()))
            project_names.append(project.getName())

    def __show_datasets__(self, project_id):
        datasets = self.conn.getObjects("Dataset", opts={'project': project_id})
        dataset_names = []
        for dataset in datasets:
            indent = " " * 5
            print(f"{indent}|--- {str(dataset.getId())} : {str(dataset.getName())}")
            dataset_names.append(dataset.getName())

    def __show_images__(self, dataset_id):
        images = self.conn.getObjects("Image", opts={'dataset': dataset_id})
        image_names = []
        for image in images:
            indent = " " * 10
            print(f"{indent}|--- {str(image.getId())} : {str(image.getName())}")
            image_names.append(image.getName())
    
    def show(self):
        self.__show_projects__()
        project_id = str(input("Type the id of the project you want to access:\n"))
        self.__show_datasets__(project_id)
        dataset_id = str(input("Type the id of the dataset you want to access:\n"))
        self.__show_images__(dataset_id)

    def get_image(self):
        omero_image_id=str(input("Type image id to open: \n"))
        img_omero_obj, img_nparray = ezomero.get_image(self.conn, int(omero_image_id))
        return img_nparray
    
    def get_dataset(self):
        dataset_id = str(input("Type the id of the dataset to open:\n"))
        images_obj = self.conn.getObjects("Image", opts={'dataset': dataset_id})
        img_nparray_list = []
        for image_obj in images_obj:
            img_omero_obj, img_nparray = ezomero.get_image(self.conn, image_obj.getId())
            img_nparray_list.append(img_nparray)
        return img_nparray_list
    
    def run(self):
        self.connect()
        command = str(input)


            
if __name__ == "__main__":
    conn = OMEROConnection()
    conn.connect()
    conn.show()
    conn.disconnect()
