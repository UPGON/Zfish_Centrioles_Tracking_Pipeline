import ezomero
import getpass
class OMEROConnection:
    def __init__(self):
        super().__init__()
        self.__host = "omero-server.epfl.ch"
        self.__port = 4064
        self.conn = None
        # track whether the connection has been explicitly closed
        self._closed = True
        self.current_image = None

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

    def __project_exists(self, project_id):
        try:
            project = self.conn.getObject("Project", int(project_id))
            return project is not None
        except (ValueError, TypeError):
            # Invalid ID format
            return False
        
    def __dataset_exists(self, dataset_id):
        try:
            dataset = self.conn.getObject("Dataset", int(dataset_id))
            return dataset is not None
        except (ValueError, TypeError):
            # Invalid ID format
            return False
        
    def __images_exists(self, image_id):
        try:
            image = self.conn.getObject("Image", int(image_id))
            return image is not None
        except (ValueError, TypeError):
            # Invalid ID format
            return False

    def __show_projects(self):
        print("List of available data in your session \n[Id: Name]")
        projects = self.conn.getObjects("Project", opts={'owner': self.conn.getUser().getId()})
        project_names = []
        for project in projects:
            print(str(project.getId())+" : "+str(project.getName()))
            project_names.append(project.getName())

    def __show_datasets(self, project_id):
        if not self.__project_exists(project_id):
            print(f"Error: Project with ID {project_id} does not exist or is not accessible.")
        else:
            datasets = self.conn.getObjects("Dataset", opts={'project': project_id})
            dataset_names = []
            for dataset in datasets:
                indent = " " * 5
                print(f"{indent}|--- {str(dataset.getId())} : {str(dataset.getName())}")
                dataset_names.append(dataset.getName())

    def __show_images(self, dataset_id):
        if not self.__dataset_exists(dataset_id):
            print(f"Error: Dataset with ID {dataset_id} does not exist or is not accessible.")
        else:
            images = self.conn.getObjects("Image", opts={'dataset': dataset_id})
            image_names = []
            for image in images:
                indent = " " * 10
                print(f"{indent}|--- {str(image.getId())} : {str(image.getName())}")
                image_names.append(image.getName())
    
    def show(self):
        self.__show_projects()
        project_id = str(input("Type the id of the project you want to access:\n"))
        self.__show_datasets(project_id)
        dataset_id = str(input("Type the id of the dataset you want to access:\n"))
        self.__show_images(dataset_id)

    def get_dataset(self):
        dataset_id = str(input("Type the id of the dataset to open:\n"))
        img_nparray_list = []
        if not self.__dataset_exists(dataset_id):
            print(f"Error: Dataset with ID {dataset_id} does not exist or is not accessible.")
            return img_nparray_list
        else:
            images_obj = self.conn.getObjects("Image", opts={'dataset': dataset_id})
            for image_obj in images_obj:
                _, img_nparray = ezomero.get_image(self.conn, image_obj.getId())
                img_nparray_list.append(img_nparray)
            return img_nparray_list

    def get_image(self):
        image_id=str(input("Type image id to open: \n"))
        if not self.__images_exists(image_id):
            print(f"Error: Image with ID {image_id} does not exist or is not accessible.")
            return []
        else:
            _, img_nparray = ezomero.get_image(self.conn, int(image_id))
            return img_nparray
        
    def connect(self):
        username=str(input("Type Your Username:\n"))
        password=getpass.getpass()

        self.conn = ezomero.connect(user=username, password=password, group="",
                            host=self.__host, port=self.__port, secure=True)

        if self.conn is not None and self.conn.isConnected():
            print(f"Connected to {self.__host}")
            self._closed = False
        else:
            print("ERROR: Not able to connect to OMERO server. Please check your credentials, group and hostname")

    def disconnect(self):
        if self.conn is not None and self.conn.isConnected():
            self.conn.close()
            self._closed = True
            print(f"Disconnect from {self.__host}")
        else:
            # silently ignore repeated disconnects, but log for debugging
            print("(disconnect called but connection already closed)")

    
    def run(self):
        self.connect()
        while True:
            command = input("Enter command (or 'quit' to exit): ").strip()
            if command.lower() == 'quit':
                break
            parts = command.split()
            if not parts:
                continue
            instruction = parts[0].lower()
            match instruction:
                case "show":
                    if(parts) == 1:
                        self.show()  # show all projects
                    elif len(parts) == 3:
                        [type, id] = parts[1:]
                        match type.upper():
                            case "P":
                                self.__show_datasets__(id)
                            case "D":
                                self.__show_images__(id)
                    else:
                        print("Usage: show [P <project_id>] or show D <dataset_id>")
                case "open":
                    if len(parts) == 3: 
                        [type, id] = parts[1:]
                        match type.upper():
                            case "D":
                                dataset = self.get_dataset(id)
                            case "I":
                                img = self.get_image(id)
                    else: 
                        print("Usage: open I <image_id> or open D <dataset_id>")
        self.disconnect()

      
if __name__ == "__main__":
    conn = OMEROConnection()
    conn.show()
    conn.disconnect()
