"""
upload_comet_images.py
Code to import any image on OMERO and attach one or more files to it.
-----------------------------------------------------------------------------
  Copyright (C) 2023
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License along
  with this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
------------------------------------------------------------------------------
Created by Rémy Dornier - EPFL - BIOP
Date: 2025.03.12
Version: 1.0.0

Dependencies
    - PyQt6
    - ezomero
"""
import os
import tempfile
import yaml
from omero.cli import CLI
import ezomero
import traceback
from PyQt6.QtWidgets import QLineEdit, QLabel, QFileDialog, QPushButton, QMainWindow, QVBoxLayout, \
    QWidget, QApplication, QHBoxLayout, QComboBox
from omero.plugins.sessions import SessionsControl
from importlib import import_module
ImportControl = import_module("omero.plugins.import").ImportControl

FONT_SIZE = 'font-size: 14px'
SEPARATOR = ","
NEW_PREFIX = "$new$"
FIXED_WIDTH = 300
HOST = "omero-server.epfl.ch"
PORT = 4064

GROUP = "group"
DST_NAME = "datasetName"
PRJ_NAME= "projectName"
FOL_PATH = "path"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.is_connected = False
        self.conn = None
        self.project_dict = {}
        self.group_dict = {}
        self.dataset_dict = {}
        self.image_dict = {}
        self.get_list = []
        self.listen_projects = True

        # main window settings
        self.setWindowTitle("Main window title")
        self.setMinimumSize(600, 200)
        widgets = []
        main_layout = QVBoxLayout()

        # username fields
        username_layout = QHBoxLayout()
        username_label = QLabel("Username")
        username_label.setStyleSheet(FONT_SIZE)
        self.username = QLineEdit()
        self.username.setStyleSheet(FONT_SIZE)
        username_widget = QWidget()
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username)
        username_widget.setLayout(username_layout)
        widgets.append(username_widget)

        # password fields
        password_layout = QHBoxLayout()
        password_label = QLabel("Password")
        password_label.setStyleSheet(FONT_SIZE)
        self.password = QLineEdit()
        self.password.setStyleSheet(FONT_SIZE)
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        password_widget = QWidget()
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password)
        password_widget.setLayout(password_layout)
        widgets.append(password_widget)

        # connection button
        self.connect_button = QPushButton(text="Connect")
        self.connect_button.setStyleSheet(FONT_SIZE)
        self.connect_button.clicked.connect(self.connect_to_omero)
        widgets.append(self.connect_button)

        # group fields
        group_layout = QHBoxLayout()
        group_label = QLabel("Group")
        group_label.setStyleSheet(FONT_SIZE)
        self.group_combo = QComboBox()
        self.group_combo.setStyleSheet(FONT_SIZE)
        self.group_combo.setEnabled(False)
        group_widget = QWidget()
        group_layout.addWidget(group_label)
        group_layout.addWidget(self.group_combo)
        group_widget.setLayout(group_layout)
        widgets.append(group_widget)

        # project fields
        project_layout = QHBoxLayout()

        project_label = QLabel("Project")
        project_label.setStyleSheet(FONT_SIZE)

        self.project_combo = QComboBox()
        self.project_combo.setEnabled(False)
        self.project_combo.setStyleSheet(FONT_SIZE)

        project_widget = QWidget()
        project_layout.addWidget(project_label)
        project_layout.addWidget(self.project_combo)
        project_widget.setLayout(project_layout)
        widgets.append(project_widget)

        # dataset fields
        dataset_layout = QHBoxLayout()

        dataset_label = QLabel("Dataset")
        dataset_label.setStyleSheet(FONT_SIZE)

        self.dataset_combo = QComboBox()
        self.dataset_combo.setEnabled(False)
        self.dataset_combo.setStyleSheet(FONT_SIZE)

        dataset_widget = QWidget()
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.dataset_combo)
        dataset_widget.setLayout(dataset_layout)
        widgets.append(dataset_widget)

        # folder fields
        folder_layout = QHBoxLayout()

        folder_label = QLabel("Image path")
        folder_label.setStyleSheet(FONT_SIZE)

        self.folder = QComboBox()
        self.folder.setEnabled(False)
        self.folder.setStyleSheet(FONT_SIZE)

        folder_widget = QWidget()
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder)
        folder_widget.setLayout(folder_layout)
        widgets.append(folder_widget)
        
        # buttons fields
        button_layout = QHBoxLayout()
        ok_button = QPushButton(text="Ok")
        ok_button.setStyleSheet(FONT_SIZE)
        ok_button.clicked.connect(self.run_app)
        next_button = QPushButton(text="Open")
        #return images
        next_button.setStyleSheet(FONT_SIZE)
        cancel_button = QPushButton(text="Cancel")
        cancel_button.setStyleSheet(FONT_SIZE)
        cancel_button.clicked.connect(self.close_app)
        button_widget = QWidget()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(next_button)
        button_layout.addWidget(cancel_button)
        button_widget.setLayout(button_layout)
        widgets.append(button_widget)

        # building the main GUI
        for w in widgets:
            main_layout.addWidget(w)

        widget = QWidget()
        widget.setLayout(main_layout)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)

    def open_file(self):
        #open file 
        rsp_path_list = QFileDialog.getOpenFileName(parent=self, caption="Select an image")

    def close_app(self):
        if self.conn is not None and self.conn.isConnected:
            self.conn.close()
        self.close()


    def run_app(self):
        username = self.username.text()
        password = self.password.text()

        self.close()
        self.conn.close()
        run_script(HOST, PORT, username, password, self.upload_list, self.project_dict, self.dataset_dict)


    def open_file_chooser(self):
        rsp_path_list = QFileDialog.getOpenFileName(parent=self, caption="Select an image")
        self.folder.setText(str(rsp_path_list))

    def connect_to_omero(self):
        username = self.username.text()
        password = self.password.text()

        self.conn = ezomero.connect(username, password, group="", host=HOST, port=PORT, secure=True)

        if self.conn is not None and self.conn.isConnected():
            self.is_connected = True
            self.folder.setEnabled(True)
            self.password.setEnabled(False)
            self.username.setEnabled(False)
            self.connect_button.setEnabled(False)
            self.group_combo.setEnabled(True)

            self.load_groups()
            self.group_combo.currentTextChanged.connect(self.group_text_changed)

            project_names = sorted(self.list_projects(self.conn))
            for project_name in project_names:
                self.project_combo.addItem(project_name)
            if len(project_names) > 0:
                self.dataset_combo.setCurrentText(project_names[0])
                self.project_text_changed(project_names[0])
                self.project_combo.setEnabled(True)


    def load_groups(self):

        if self.conn is not None and self.conn.isConnected():
            group_names = sorted(self.list_groups(self.conn))
            for group_name in group_names:
                self.group_combo.addItem(group_name)

            if len(group_names) > 0:
                group_name = self.conn.getEventContext().groupName
                self.group_combo.setCurrentText(group_name)
                self.connect_button.setEnabled(True)
            else: 
                self.dataset_combo.setCurrentText("No project found")


    def list_groups(self, conn):
        # Retrieve the services we are going to use
        admin_service = conn.getAdminService()

        ec = admin_service.getEventContext()
        groups = [admin_service.getGroup(v) for v in ec.memberOfGroups]
        group_names = []
        for group in groups:
            if group.id.val in [0, 1, 2]:
                continue
            group_names.append(group.name.val)
            self.group_dict[group.name.val] = group.id.val
        return group_names


    def list_projects(self, conn):
        projects = conn.getObjects("Project", opts={'owner': conn.getUser().getId()})
        project_names = []
        for project in projects:
            project_names.append(project.getName())
            self.project_dict[project.getName()] = project.getId()
        return project_names


    def list_datasets(self, conn, project_id):
        project = conn.getObject("Project", project_id)
        dataset_names = []
        for dataset in project.listChildren():
            dataset_names.append(dataset.getName())
            self.dataset_dict[dataset.getName()] = dataset.getId()
        return dataset_names
    
    def list_images(self, conn, dataset_id):
        dataset = conn.getObject("Dataset", dataset_id)
        image_names = []
        for image in dataset.listChildren():
            image_names.append(image.getName())
        return image_names

    def group_text_changed(self):
        group_name = self.group_combo.currentText()
        self.conn.SERVICE_OPTS.setOmeroGroup(self.group_dict[group_name])

        project_names = sorted(self.list_projects(self.conn))
        self.listen_projects = False
        self.project_combo.clear()
        for project_name in project_names:
            self.project_combo.addItem(project_name)
        if len(project_names) > 0:
            self.project_combo.setCurrentText(project_names[0])
            self.listen_projects = True
            self.project_text_changed(project_names[0])


    def project_text_changed(self, project_name):
        if self.listen_projects:
            dataset_names = sorted(self.list_datasets(self.conn, self.project_dict[project_name]))
            self.dataset_combo.clear()
            for dataset_name in dataset_names:
                self.dataset_combo.addItem(dataset_name)
            if len(dataset_names) > 0:
                self.dataset_combo.setCurrentText(dataset_names[0])
                self.dataset_combo.setEnabled(True)
                self.dataset_text_changed(dataset_names[0])
            else: 
                self.dataset_combo.setCurrentText("No dataset found")

    def dataset_text_changed(self, dataset_name):
        image_names = sorted(self.list_images(self.conn, self.dataset_dict[dataset_name]))
        self.folder.clear()
        for image_name in image_names:
            self.folder.addItem(image_name)
        if len(image_names) > 0:
            self.folder.setCurrentText(image_names[0])
            self.folder.setEnabled(True)
        else: 
            self.folder.setCurrentText("No image found")


def extract_image_id(fname):
    """Parse the YAML returned by an 'omero import' call and extract the image ID.

    Parameters
    ----------
    fname : str
        The path to the `yaml` file to parse.

    Returns
    -------
    int or None
        The OMERO ID of the newly imported image, e.g. `1568386` or `None` in case
        parsing the file failed for any reason.
    """

    try:
        with open(fname, "r", encoding="utf-8") as stream:
            parsed = yaml.safe_load(stream)
        if len(parsed[0]["Image"]) != 1:
            if parsed[0]["Fileset"] is not None:
                image_id = ",".join([str(img_id) for img_id in parsed[0]["Image"]])
            else:
                msg = f"Unexpected YAML retrieved from OMERO, unable to parse:\n{parsed}"
                print("ERROR", msg)
                raise SyntaxError(msg)
        else:
            image_id = parsed[0]["Image"][0]
    except Exception as err:  # pylint: disable-msg=broad-except
        print("ERROR", f"Error parsing imported image ID from YAML output: {err}")
        return None

    print(f"Successfully parsed Image ID from YAML: {image_id}")
    return str(image_id)



def run_script(host, port, username, password, upload_task_list, project_dict, dataset_dict):

    cli = CLI()
    cli.register('import', ImportControl, '_')
    cli.register('sessions', SessionsControl, '_')

    for upload_task in upload_task_list:
        project_name = upload_task[PRJ_NAME]
        dataset_name = upload_task[DST_NAME]
        group = upload_task[GROUP]


        conn = ezomero.connect(username, password, group=group, host=host, port=port, secure=True)

        if conn is not None and conn.isConnected():
            print(f"Connected to {host}")  

            try:
                if project_name is not None and project_name != "":
                    project_id = project_dict[project_name]

                    if dataset_name is not None and dataset_name != "":
                        dataset_id = dataset_dict[dataset_name]

                        try:
                            imgs_omero = []
                            # obtaining the image in the right dataset
                            for idx, omero_image_id in enumerate(dataset_id):
                                img_omero_obj, img_nparray = ezomero.get_image(conn, int(omero_image_id))
                                imgs_omero.append(img_omero_obj)

                            print("The image "+str(img_omero_obj.getId())+" has been imported from OMERO")
                            return imgs_omero
                        except Exception as e:
                            print(e)
                            traceback.print_exc()
                        finally:
                            conn.close()
                            print(f"Disconnect from {host}")

                        # because the upload can be long, we need to re-connect again to omero
                        if conn is None or not conn.isConnected():
                            print(f"Reconnection to {host}...")
                            conn = ezomero.connect(username, password, group=group, host=host, port=port, secure=True)
                            print(f"Reconnected...")
                    else:
                        print("Give a valid name to the dataset !")
                else:
                    print("Give a valid name to the project !")

            except Exception as e:
                print(e)
                traceback.print_exc()
            finally:
                conn.close()
                print(f"Disconnect from {host}")



if __name__ == "__main__":
    list_argv = []
    app = QApplication(list_argv)
    window = MainWindow()
    window.show()
    app.exec()
