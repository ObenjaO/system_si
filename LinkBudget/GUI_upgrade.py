import sys
import sqlite3
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QGraphicsView, QGraphicsScene,
                             QGroupBox, QComboBox, QCheckBox, QMessageBox)
from PyQt6.QtGui import QPen, QColor, QBrush, QFont, QLinearGradient, QPixmap
from PyQt6.QtCore import Qt


class LinkBudgetWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rack Link Budget Calculator")
        self.setGeometry(100, 100, 1400, 1000)

        # Insertion Loss Table (in dB)
        self.insertion_loss_table = {
            "PCI": {
                "PCI-Gen6": 32,
                "PCI-Gen5": 36,
                "PCI-Gen4": 28,
                "PCI-Gen3": 23.5
            },
            "KR1": {  # Ball-to-ball
                "50GBASE": 28,
                "100GBASE": 28,
                "200GBASE": 28 # TBD 
            },
            "CR1": {
                "50GBASE": 5.45,
                "100GBASE": 6.875,
                "200GBASE": 6.875 # TBD 
            },
            "C2M": {
                "50GBASE": 7.5,
                "100GBASE": 11.9,
                "200GBASE": 11.9 # TBD 
            }
        }

        self.block_width = 70
        self.block_height = 35
        self.via_ellipse_size = 15
        self.via_small_ellipse_size = 5
        self.delta_via = (self.via_ellipse_size - self.via_small_ellipse_size) // 2

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        bold_font = QFont()
        bold_font.setBold(True)

        frames_layout = QHBoxLayout()

        # ACC Input GroupBox
        input_group = QGroupBox("ACC Input")
        input_group.setStyleSheet("QGroupBox { border: 2px solid gray; border-radius: 5px; margin-top: 5px; background-color: #f0f0f0; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 2px; }")
        input_group.setFixedWidth(600)
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(2)
        input_layout.setContentsMargins(5, 5, 5, 5)

        self.ip_label = QLabel("IP Loss (dB):")
        self.ip_label.setFont(bold_font)
        self.ip_input = QLineEdit("0.0")
        self.ip_input.setFixedWidth(40)
        input_layout.addWidget(self.ip_label)
        input_layout.addWidget(self.ip_input)

        self.package_loss_per_mm_label = QLabel("Package")
        self.package_loss_per_mm_label.setFont(bold_font)
        input_layout.addWidget(self.package_loss_per_mm_label)
        package_h_layout = QHBoxLayout()
        self.package_loss_label = QLabel("Loss[dB/mm]")
        self.package_loss_per_mm_input = QLineEdit("-0.15")
        self.package_loss_per_mm_input.setFixedWidth(40)
        package_h_layout.addWidget(self.package_loss_label)
        package_h_layout.addWidget(self.package_loss_per_mm_input)
        self.package_length_label = QLabel("Length[mm]")
        self.package_length_input = QLineEdit("2")
        self.package_length_input.setFixedWidth(40)
        package_h_layout.addWidget(self.package_length_label)
        package_h_layout.addWidget(self.package_length_input)
        package_h_layout.addStretch()
        input_layout.addLayout(package_h_layout)

        self.via_label = QLabel("VIA Losses (dB):")
        self.via_label.setFont(bold_font)
        input_layout.addWidget(self.via_label)
        via_h_layout = QHBoxLayout()
        self.via_breakout_label = QLabel("Breakout:")
        self.via_breakout_input = QLineEdit("-2")
        self.via_breakout_input.setFixedWidth(40)
        via_h_layout.addWidget(self.via_breakout_label)
        via_h_layout.addWidget(self.via_breakout_input)
        self.via_in_label = QLabel("In:")
        self.via_in_input = QLineEdit("-2")
        self.via_in_input.setFixedWidth(40)
        via_h_layout.addWidget(self.via_in_label)
        via_h_layout.addWidget(self.via_in_input)
        self.via_out_label = QLabel("Out:")
        self.via_out_input = QLineEdit("-2")
        self.via_out_input.setFixedWidth(40)
        via_h_layout.addWidget(self.via_out_label)
        via_h_layout.addWidget(self.via_out_input)
        self.via_rtin_label = QLabel("RTin:")
        self.via_rtin_input = QLineEdit("-2")
        self.via_rtin_input.setFixedWidth(40)
        via_h_layout.addWidget(self.via_rtin_label)
        via_h_layout.addWidget(self.via_rtin_input)
        self.via_rtout_label = QLabel("RTout:")
        self.via_rtout_input = QLineEdit("-2")
        self.via_rtout_input.setFixedWidth(40)
        via_h_layout.addWidget(self.via_rtout_label)
        via_h_layout.addWidget(self.via_rtout_input)
        input_layout.addLayout(via_h_layout)

        self.pcb_card_loss_per_inch_label = QLabel("PCB_card")
        self.pcb_card_loss_per_inch_label.setFont(bold_font)
        input_layout.addWidget(self.pcb_card_loss_per_inch_label)
        pcb_card_h_layout = QHBoxLayout()
        self.pcb_card_loss_label = QLabel("Loss[dB/inch]")
        self.pcb_card_loss_per_inch_input = QLineEdit("-1")
        self.pcb_card_loss_per_inch_input.setFixedWidth(40)
        pcb_card_h_layout.addWidget(self.pcb_card_loss_label)
        pcb_card_h_layout.addWidget(self.pcb_card_loss_per_inch_input)
        self.pcb_card_length_label = QLabel("Length[inch]")
        self.pcb_card_length_input = QLineEdit("3")
        self.pcb_card_length_input.setFixedWidth(40)
        pcb_card_h_layout.addWidget(self.pcb_card_length_label)
        pcb_card_h_layout.addWidget(self.pcb_card_length_input)
        self.pcb_card_material_combo = QComboBox()
        self.pcb_card_material_combo.addItems(["manual", "FR-4", "Rogers RO4003C", "Rogers RO4350B", "Isola I-Tera MT40", "Nelco N4000-13", "Teflon (PTFE)", "Megtron 6"])
        self.pcb_card_material_combo.setCurrentText("manual")
        self.pcb_card_material_combo.setFixedWidth(100)
        self.pcb_card_material_combo.setStyleSheet("QComboBox { background-color: #D9E2A6; }")
        self.pcb_card_material_combo.currentTextChanged.connect(self.update_pcb_card_loss)
        pcb_card_h_layout.addWidget(self.pcb_card_material_combo)
        pcb_card_h_layout.addStretch()
        input_layout.addLayout(pcb_card_h_layout)

        self.b2b_con_label = QLabel("B2B_con Loss (dB):")
        self.b2b_con_label.setFont(bold_font)
        self.b2b_con_input = QLineEdit("0.0")
        self.b2b_con_input.setFixedWidth(40)
        input_layout.addWidget(self.b2b_con_label)
        input_layout.addWidget(self.b2b_con_input)

        self.pcb_main_loss_per_inch_label = QLabel("PCB_main")
        self.pcb_main_loss_per_inch_label.setFont(bold_font)
        input_layout.addWidget(self.pcb_main_loss_per_inch_label)
        pcb_main_h_layout = QHBoxLayout()
        self.pcb_main_loss_label = QLabel("Loss[dB/inch]")
        self.pcb_main_loss_per_inch_input = QLineEdit("-1")
        self.pcb_main_loss_per_inch_input.setFixedWidth(40)
        pcb_main_h_layout.addWidget(self.pcb_main_loss_label)
        pcb_main_h_layout.addWidget(self.pcb_main_loss_per_inch_input)
        self.pcb_main_length_label = QLabel("Length[inch]")
        self.pcb_main_length_input = QLineEdit("3")
        self.pcb_main_length_input.setFixedWidth(40)
        pcb_main_h_layout.addWidget(self.pcb_main_length_label)
        pcb_main_h_layout.addWidget(self.pcb_main_length_input)
        self.pcb_main_material_combo = QComboBox()
        self.pcb_main_material_combo.addItems(["manual", "FR-4", "Rogers RO4003C", "Rogers RO4350B", "Isola I-Tera MT40", "Nelco N4000-13", "Teflon (PTFE)", "Megtron 6"])
        self.pcb_main_material_combo.setCurrentText("manual")
        self.pcb_main_material_combo.setFixedWidth(100)
        self.pcb_main_material_combo.setStyleSheet("QComboBox { background-color: #D9E2A6; }")
        self.pcb_main_material_combo.currentTextChanged.connect(self.update_pcb_main_loss)
        pcb_main_h_layout.addWidget(self.pcb_main_material_combo)
        pcb_main_h_layout.addStretch()
        input_layout.addLayout(pcb_main_h_layout)

        self.rt_label = QLabel("RT Loss (dB):")
        self.rt_label.setFont(bold_font)
        self.rt_input = QLineEdit("-2")
        self.rt_input.setFixedWidth(40)
        input_layout.addWidget(self.rt_label)
        input_layout.addWidget(self.rt_input)

        self.pcb_main2_loss_per_inch_label = QLabel("PCB_main2")
        self.pcb_main2_loss_per_inch_label.setFont(bold_font)
        input_layout.addWidget(self.pcb_main2_loss_per_inch_label)
        pcb_main2_h_layout = QHBoxLayout()
        self.pcb_main2_loss_label = QLabel("Loss[dB/inch]")
        self.pcb_main2_loss_per_inch_input = QLineEdit("-1")
        self.pcb_main2_loss_per_inch_input.setFixedWidth(40)
        pcb_main2_h_layout.addWidget(self.pcb_main2_loss_label)
        pcb_main2_h_layout.addWidget(self.pcb_main2_loss_per_inch_input)
        self.pcb_main2_length_label = QLabel("Length[inch]")
        self.pcb_main2_length_input = QLineEdit("3")
        self.pcb_main2_length_input.setFixedWidth(40)
        pcb_main2_h_layout.addWidget(self.pcb_main2_length_label)
        pcb_main2_h_layout.addWidget(self.pcb_main2_length_input)
        self.pcb_main2_material_combo = QComboBox()
        self.pcb_main2_material_combo.addItems(["manual", "FR-4", "Rogers RO4003C", "Rogers RO4350B", "Isola I-Tera MT40", "Nelco N4000-13", "Teflon (PTFE)", "Megtron 6"])
        self.pcb_main2_material_combo.setCurrentText("manual")
        self.pcb_main2_material_combo.setFixedWidth(100)
        self.pcb_main2_material_combo.setStyleSheet("QComboBox { background-color: #D9E2A6; }")
        self.pcb_main2_material_combo.currentTextChanged.connect(self.update_pcb_main2_loss)
        pcb_main2_h_layout.addWidget(self.pcb_main2_material_combo)
        pcb_main2_h_layout.addStretch()
        input_layout.addLayout(pcb_main2_h_layout)

        self.connector_label = QLabel("Connector Loss (dB):")
        self.connector_label.setFont(bold_font)
        self.connector_input = QLineEdit("-2.5")
        self.connector_input.setFixedWidth(40)
        input_layout.addWidget(self.connector_label)
        input_layout.addWidget(self.connector_input)

        self.cable_loss_per_meter_label = QLabel("Cable")
        self.cable_loss_per_meter_label.setFont(bold_font)
        input_layout.addWidget(self.cable_loss_per_meter_label)
        cable_h_layout = QHBoxLayout()
        self.cable_loss_label = QLabel("Loss[dB/meter]")
        self.cable_loss_per_meter_input = QLineEdit("0.0")
        self.cable_loss_per_meter_input.setFixedWidth(40)
        cable_h_layout.addWidget(self.cable_loss_label)
        cable_h_layout.addWidget(self.cable_loss_per_meter_input)
        self.cable_length_label = QLabel("Length[meter]")
        self.cable_length_input = QLineEdit("1")
        self.cable_length_input.setFixedWidth(40)
        cable_h_layout.addWidget(self.cable_length_label)
        cable_h_layout.addWidget(self.cable_length_input)
        self.cable_material_combo = QComboBox()
        self.cable_material_combo.addItems(["manual", "twinax_AWG26","twinax_AWG28","twinax_AWG30","twinax_AWG32","AWG 24", "AWG 26", "AWG 28", "AWG 30",
                                            "DAC AWG 24", "DAC AWG 26", "DAC AWG 28", "DAC AWG 30"])
        self.cable_material_combo.setCurrentText("manual")
        self.cable_material_combo.setFixedWidth(100)
        self.cable_material_combo.setStyleSheet("QComboBox { background-color: #C6CDB9; }")
        self.cable_material_combo.currentTextChanged.connect(self.update_cable_loss)
        cable_h_layout.addWidget(self.cable_material_combo)
        cable_h_layout.addStretch()
        input_layout.addLayout(cable_h_layout)

        frames_layout.addWidget(input_group)

        # Interface GroupBox
        interface_group = QGroupBox("Interface")
        interface_group.setStyleSheet("QGroupBox { border: 2px solid gray; border-radius: 5px; margin-top: 5px; background-color: #f0f0f0; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 2px; }")
        interface_group.setFixedWidth(150)
        interface_layout = QVBoxLayout(interface_group)
        interface_layout.setSpacing(8)
        interface_layout.setContentsMargins(5, 5, 5, 5)

        self.baud_rate_label = QLabel("data rate[Gps]:")
        self.baud_rate_label.setFont(bold_font)
        self.baud_rate_input = QLineEdit("1.0")
        self.baud_rate_input.setFixedWidth(60)
        self.baud_rate_input.textChanged.connect(self.update_all_material_losses)
        interface_layout.addWidget(self.baud_rate_label)
        interface_layout.addWidget(self.baud_rate_input)

        self.signaling_label = QLabel("Signaling:")
        self.signaling_label.setFont(bold_font)
        self.signaling_input = QComboBox()
        self.signaling_input.addItems(["PAM2", "PAM4"])
        self.signaling_input.setFixedWidth(80)
        interface_layout.addWidget(self.signaling_label)
        interface_layout.addWidget(self.signaling_input)

        self.b2b_en = QCheckBox("B2B_en")
        self.b2b_en.setFont(bold_font)
        self.b2b_en.setChecked(True)
        interface_layout.addWidget(self.b2b_en)
        self.b2b_en.stateChanged.connect(self.redraw_cascade)

        self.rt_en = QCheckBox("RT_en")
        self.rt_en.setFont(bold_font)
        self.rt_en.setChecked(True)
        interface_layout.addWidget(self.rt_en)
        self.rt_en.stateChanged.connect(self.redraw_cascade)

        self.sw_rt_en = QCheckBox("sw_RT_en")
        self.sw_rt_en.setFont(bold_font)
        self.sw_rt_en.setChecked(True)
        interface_layout.addWidget(self.sw_rt_en)
        self.sw_rt_en.stateChanged.connect(self.redraw_cascade)

        self.sw_b2b_en = QCheckBox("sw_B2B_en")
        self.sw_b2b_en.setFont(bold_font)
        self.sw_b2b_en.setChecked(True)
        interface_layout.addWidget(self.sw_b2b_en)
        self.sw_b2b_en.stateChanged.connect(self.redraw_cascade)
        #interface_layout.addStretch()

        self.con_en = QCheckBox("connector_en")
        self.con_en.setFont(bold_font)
        self.con_en.setChecked(True)
        interface_layout.addWidget(self.con_en)
        self.con_en.stateChanged.connect(self.redraw_cascade)
        interface_layout.addStretch()
        #frames_layout.addWidget(interface_group)

        # Add logo below Interface GroupBox with smaller size
        logo_label = QLabel(self)
        logo_file = "logo.jpg"  # Ensure this matches your exact file name

        # Determine the base path
        if getattr(sys, 'frozen', False):  # Running as a PyInstaller executable
            base_path = sys._MEIPASS
        else:  # Running as a script
            base_path = os.path.dirname(os.path.abspath(__file__))

        # Construct the logo path
        logo_path = os.path.join(base_path, logo_file)
        print(f"Debug: Attempting to load logo from: {logo_path}")
        print(f"Debug: Does file exist? {os.path.exists(logo_path)}")

        # Load the logo
        logo_pixmap = QPixmap(logo_path)
        if logo_pixmap.isNull():
            # Fallback: Try the current working directory
            logo_path = os.path.join(os.getcwd(), logo_file)
            print(f"Debug: Fallback to current working directory: {logo_path}")
            print(f"Debug: Does fallback file exist? {os.path.exists(logo_path)}")
            logo_pixmap = QPixmap(logo_path)

        # Check if the logo loaded successfully
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)  # Reduced to 50x50
            logo_label.setPixmap(logo_pixmap)
        else:
            error_msg = f"Logo not loaded. Tried paths:\n{os.path.join(base_path, logo_file)}\n{logo_path}"
            print(error_msg)
            logo_label.setText("Logo not loaded (check file path)")
            QMessageBox.warning(self, "Logo Error", error_msg)

        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        interface_layout.addWidget(logo_label)  # Add to interface_layout
        frames_layout.addWidget(interface_group)  # Add to frames_layout (same layout as interface_group)

        # Switch Frame GroupBox
        switch_group = QGroupBox("Switch Frame")
        switch_group.setStyleSheet("QGroupBox { border: 2px solid gray; border-radius: 5px; margin-top: 5px; background-color: #f0f0f0; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 2px; }")
        switch_group.setFixedWidth(600)
        switch_layout = QVBoxLayout(switch_group)
        switch_layout.setSpacing(2)
        switch_layout.setContentsMargins(5, 5, 5, 5)

        self.sw_ip_label = QLabel("IP Loss (dB):")
        self.sw_ip_label.setFont(bold_font)
        self.sw_ip_input = QLineEdit("0.0")
        self.sw_ip_input.setFixedWidth(40)
        switch_layout.addWidget(self.sw_ip_label)
        switch_layout.addWidget(self.sw_ip_input)

        self.sw_package_loss_per_mm_label = QLabel("Package")
        self.sw_package_loss_per_mm_label.setFont(bold_font)
        switch_layout.addWidget(self.sw_package_loss_per_mm_label)
        sw_package_h_layout = QHBoxLayout()
        self.sw_package_loss_label = QLabel("Loss[dB/mm]")
        self.sw_package_loss_per_mm_input = QLineEdit("-0.15")
        self.sw_package_loss_per_mm_input.setFixedWidth(40)
        sw_package_h_layout.addWidget(self.sw_package_loss_label)
        sw_package_h_layout.addWidget(self.sw_package_loss_per_mm_input)
        self.sw_package_length_label = QLabel("Length[mm]")
        self.sw_package_length_input = QLineEdit("2")
        self.sw_package_length_input.setFixedWidth(40)
        sw_package_h_layout.addWidget(self.sw_package_length_label)
        sw_package_h_layout.addWidget(self.sw_package_length_input)
        sw_package_h_layout.addStretch()
        switch_layout.addLayout(sw_package_h_layout)

        self.sw_via_label = QLabel("VIA Losses (dB):")
        self.sw_via_label.setFont(bold_font)
        switch_layout.addWidget(self.sw_via_label)
        sw_via_h_layout = QHBoxLayout()
        self.sw_via_breakout_label = QLabel("Breakout:")
        self.sw_via_breakout_input = QLineEdit("-2")
        self.sw_via_breakout_input.setFixedWidth(40)
        sw_via_h_layout.addWidget(self.sw_via_breakout_label)
        sw_via_h_layout.addWidget(self.sw_via_breakout_input)
        self.sw_via_in_label = QLabel("In:")
        self.sw_via_in_input = QLineEdit("-2")
        self.sw_via_in_input.setFixedWidth(40)
        sw_via_h_layout.addWidget(self.sw_via_in_label)
        sw_via_h_layout.addWidget(self.sw_via_in_input)
        self.sw_via_out_label = QLabel("Out:")
        self.sw_via_out_input = QLineEdit("-2")
        self.sw_via_out_input.setFixedWidth(40)
        sw_via_h_layout.addWidget(self.sw_via_out_label)
        sw_via_h_layout.addWidget(self.sw_via_out_input)
        self.sw_via_rtin_label = QLabel("RTin:")
        self.sw_via_rtin_input = QLineEdit("-2")
        self.sw_via_rtin_input.setFixedWidth(40)
        sw_via_h_layout.addWidget(self.sw_via_rtin_label)
        sw_via_h_layout.addWidget(self.sw_via_rtin_input)
        self.sw_via_rtout_label = QLabel("RTout:")
        self.sw_via_rtout_input = QLineEdit("-2")
        self.sw_via_rtout_input.setFixedWidth(40)
        sw_via_h_layout.addWidget(self.sw_via_rtout_label)
        sw_via_h_layout.addWidget(self.sw_via_rtout_input)
        switch_layout.addLayout(sw_via_h_layout)

        self.sw_pcb_card_loss_per_inch_label = QLabel("PCB_card")
        self.sw_pcb_card_loss_per_inch_label.setFont(bold_font)
        switch_layout.addWidget(self.sw_pcb_card_loss_per_inch_label)
        sw_pcb_card_h_layout = QHBoxLayout()
        self.sw_pcb_card_loss_label = QLabel("Loss[dB/inch]")
        self.sw_pcb_card_loss_per_inch_input = QLineEdit("-1")
        self.sw_pcb_card_loss_per_inch_input.setFixedWidth(40)
        sw_pcb_card_h_layout.addWidget(self.sw_pcb_card_loss_label)
        sw_pcb_card_h_layout.addWidget(self.sw_pcb_card_loss_per_inch_input)
        self.sw_pcb_card_length_label = QLabel("Length[inch]")
        self.sw_pcb_card_length_input = QLineEdit("3")
        self.sw_pcb_card_length_input.setFixedWidth(40)
        sw_pcb_card_h_layout.addWidget(self.sw_pcb_card_length_label)
        sw_pcb_card_h_layout.addWidget(self.sw_pcb_card_length_input)
        self.sw_pcb_card_material_combo = QComboBox()
        self.sw_pcb_card_material_combo.addItems(["manual", "FR-4", "Rogers RO4003C", "Rogers RO4350B", "Isola I-Tera MT40", "Nelco N4000-13", "Teflon (PTFE)", "Megtron 6"])
        self.sw_pcb_card_material_combo.setCurrentText("manual")
        self.sw_pcb_card_material_combo.setFixedWidth(100)
        self.sw_pcb_card_material_combo.setStyleSheet("QComboBox { background-color: #D9E2A6; }")
        self.sw_pcb_card_material_combo.currentTextChanged.connect(self.update_sw_pcb_card_loss)
        sw_pcb_card_h_layout.addWidget(self.sw_pcb_card_material_combo)
        sw_pcb_card_h_layout.addStretch()
        switch_layout.addLayout(sw_pcb_card_h_layout)

        self.sw_b2b_con_label = QLabel("B2B_con Loss (dB):")
        self.sw_b2b_con_label.setFont(bold_font)
        self.sw_b2b_con_input = QLineEdit("0.0")
        self.sw_b2b_con_input.setFixedWidth(40)
        switch_layout.addWidget(self.sw_b2b_con_label)
        switch_layout.addWidget(self.sw_b2b_con_input)

        self.sw_pcb_main_loss_per_inch_label = QLabel("PCB_main")
        self.sw_pcb_main_loss_per_inch_label.setFont(bold_font)
        switch_layout.addWidget(self.sw_pcb_main_loss_per_inch_label)
        sw_pcb_main_h_layout = QHBoxLayout()
        self.sw_pcb_main_loss_label = QLabel("Loss[dB/inch]")
        self.sw_pcb_main_loss_per_inch_input = QLineEdit("-1")
        self.sw_pcb_main_loss_per_inch_input.setFixedWidth(40)
        sw_pcb_main_h_layout.addWidget(self.sw_pcb_main_loss_label)
        sw_pcb_main_h_layout.addWidget(self.sw_pcb_main_loss_per_inch_input)
        self.sw_pcb_main_length_label = QLabel("Length[inch]")
        self.sw_pcb_main_length_input = QLineEdit("3")
        self.sw_pcb_main_length_input.setFixedWidth(40)
        sw_pcb_main_h_layout.addWidget(self.sw_pcb_main_length_label)
        sw_pcb_main_h_layout.addWidget(self.sw_pcb_main_length_input)
        self.sw_pcb_main_material_combo = QComboBox()
        self.sw_pcb_main_material_combo.addItems(["manual", "FR-4", "Rogers RO4003C", "Rogers RO4350B", "Isola I-Tera MT40", "Nelco N4000-13", "Teflon (PTFE)", "Megtron 6"])
        self.sw_pcb_main_material_combo.setCurrentText("manual")
        self.sw_pcb_main_material_combo.setFixedWidth(100)
        self.sw_pcb_main_material_combo.setStyleSheet("QComboBox { background-color: #D9E2A6; }")
        self.sw_pcb_main_material_combo.currentTextChanged.connect(self.update_sw_pcb_main_loss)
        sw_pcb_main_h_layout.addWidget(self.sw_pcb_main_material_combo)
        sw_pcb_main_h_layout.addStretch()
        switch_layout.addLayout(sw_pcb_main_h_layout)

        self.sw_rt_label = QLabel("RT Loss (dB):")
        self.sw_rt_label.setFont(bold_font)
        self.sw_rt_input = QLineEdit("-2")
        self.sw_rt_input.setFixedWidth(40)
        switch_layout.addWidget(self.sw_rt_label)
        switch_layout.addWidget(self.sw_rt_input)

        self.sw_pcb_main2_loss_per_inch_label = QLabel("PCB_main2")
        self.sw_pcb_main2_loss_per_inch_label.setFont(bold_font)
        switch_layout.addWidget(self.sw_pcb_main2_loss_per_inch_label)
        sw_pcb_main2_h_layout = QHBoxLayout()
        self.sw_pcb_main2_loss_label = QLabel("Loss[dB/inch]")
        self.sw_pcb_main2_loss_per_inch_input = QLineEdit("-1")
        self.sw_pcb_main2_loss_per_inch_input.setFixedWidth(40)
        sw_pcb_main2_h_layout.addWidget(self.sw_pcb_main2_loss_label)
        sw_pcb_main2_h_layout.addWidget(self.sw_pcb_main2_loss_per_inch_input)
        self.sw_pcb_main2_length_label = QLabel("Length[inch]")
        self.sw_pcb_main2_length_input = QLineEdit("3")
        self.sw_pcb_main2_length_input.setFixedWidth(40)
        sw_pcb_main2_h_layout.addWidget(self.sw_pcb_main2_length_label)
        sw_pcb_main2_h_layout.addWidget(self.sw_pcb_main2_length_input)
        self.sw_pcb_main2_material_combo = QComboBox()
        self.sw_pcb_main2_material_combo.addItems(["manual", "FR-4", "Rogers RO4003C", "Rogers RO4350B", "Isola I-Tera MT40", "Nelco N4000-13", "Teflon (PTFE)", "Megtron 6"])
        self.sw_pcb_main2_material_combo.setCurrentText("manual")
        self.sw_pcb_main2_material_combo.setFixedWidth(100)
        self.sw_pcb_main2_material_combo.currentTextChanged.connect(self.update_sw_pcb_main2_loss)
        self.sw_pcb_main2_material_combo.setStyleSheet("QComboBox { background-color: #D9E2A6; }")
        sw_pcb_main2_h_layout.addWidget(self.sw_pcb_main2_material_combo)
        sw_pcb_main2_h_layout.addStretch()
        switch_layout.addLayout(sw_pcb_main2_h_layout)

        self.sw_connector_label = QLabel("Connector Loss (dB):")
        self.sw_connector_label.setFont(bold_font)
        self.sw_connector_input = QLineEdit("-2.5")
        self.sw_connector_input.setFixedWidth(40)
        switch_layout.addWidget(self.sw_connector_label)
        switch_layout.addWidget(self.sw_connector_input)
        
        frames_layout.addWidget(switch_group)
        frames_layout.addStretch()
        layout.addLayout(frames_layout)

        # Buttons and Result Label
        # In __init__, under "Buttons and Result Label" section, ensure this line exists as is:
        self.result_label = QLabel("Result will appear here")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.result_label)  # Assuming it's added to the main layout

        button_layout = QHBoxLayout()
        self.calc_button = QPushButton("Calculate")
        self.calc_button.clicked.connect(self.calculate_budget)
        self.calc_button.setFixedWidth(100)
        font = QFont()
        font.setPointSize(font.pointSize() + 2)
        self.calc_button.setFont(font)
        self.calc_button.setStyleSheet("QPushButton { background-color: navy; color: white; padding: 2px; }")
        button_layout.addWidget(self.calc_button)

        self.spec_combo = QComboBox()
        self.spec_combo.addItems(["No spec", "PCI-Gen6", "PCI-Gen5", "PCI-Gen4", "PCI-Gen3", "50GBASE", "100GBASE", "200GBASE"])
        self.spec_combo.setCurrentText("No spec")
        self.spec_combo.setFixedWidth(100)
        self.spec_combo.setFont(font)
        self.spec_combo.setStyleSheet("QComboBox { background-color: darkgreen; color: white; padding: 2px; }")
        self.spec_combo.currentTextChanged.connect(self.update_spec_settings)
        button_layout.addWidget(self.spec_combo)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        

        self.result_label = QLabel("Total Loss: 0.0 dB")
        layout.addWidget(self.result_label)

        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.graphics_view.setMinimumHeight(200)
        layout.addWidget(self.graphics_view)

        self.draw_cascade(0.0, -0.15 * 2, -2, -2, -2, -2, -2, -1 * 3, 0.0, -1 * 3, -2, -1 * 3, -2.5, 0.0 * 1,
                          0.0, -0.15 * 2, -2, -2, -2, -2, -2, -1 * 3, 0.0, -1 * 3, -2, -1 * 3, -2.5)

    # Interpolation and Extrapolation Logic
    def get_loss_from_db(self, material, target_freq):
        conn = sqlite3.connect("materials.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT l.frequency_GHz, l.loss_dB_inch 
            FROM losses l
            JOIN materials m ON l.material_id = m.material_id
            WHERE m.material_name = ?
            ORDER BY l.frequency_GHz
        """, (material,))
        data = cursor.fetchall()
        conn.close()

        if not data:
            print(f"Debug: No data found for material {material}")
            return 1.0

        frequencies, losses = zip(*data)
        frequencies = list(frequencies)
        losses = list(losses)

        if target_freq <= frequencies[0]:
            x1, x2 = frequencies[0], frequencies[1]
            y1, y2 = losses[0], losses[1]
            slope = (y2 - y1) / (x2 - x1)
            return y1 + slope * (target_freq - x1)

        if target_freq >= frequencies[-1]:
            x1, x2 = frequencies[-2], frequencies[-1]
            y1, y2 = losses[-2], losses[-1]
            slope = (y2 - y1) / (x2 - x1)
            return y2 + slope * (target_freq - x2)

        for i in range(len(frequencies) - 1):
            if frequencies[i] <= target_freq <= frequencies[i + 1]:
                x1, x2 = frequencies[i], frequencies[i + 1]
                y1, y2 = losses[i], losses[i + 1]
                return y1 + (target_freq - x1) * (y2 - y1) / (x2 - x1)
        return losses[0]

    def get_cable_loss_from_db(self, cable, target_freq):
        conn = sqlite3.connect("materials.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cl.frequency_GHz, cl.loss_dB_meter 
            FROM cable_losses cl
            JOIN cables c ON cl.cable_id = c.cable_id
            WHERE c.cable_name = ?
            ORDER BY cl.frequency_GHz
        """, (cable,))
        data = cursor.fetchall()
        conn.close()

        if not data:
            print(f"Debug: No data found for cable {cable}")
            return 1.0

        frequencies, losses = zip(*data)
        frequencies = list(frequencies)
        losses = list(losses)

        if target_freq <= frequencies[0]:
            x1, x2 = frequencies[0], frequencies[1]
            y1, y2 = losses[0], losses[1]
            slope = (y2 - y1) / (x2 - x1)
            return y1 + slope * (target_freq - x1)

        if target_freq >= frequencies[-1]:
            x1, x2 = frequencies[-2], frequencies[-1]
            y1, y2 = losses[-2], losses[-1]
            slope = (y2 - y1) / (x2 - x1)
            return y2 + slope * (target_freq - x2)

        for i in range(len(frequencies) - 1):
            if frequencies[i] <= target_freq <= frequencies[i + 1]:
                x1, x2 = frequencies[i], frequencies[i + 1]
                y1, y2 = losses[i], losses[i + 1]
                return y1 + (target_freq - x1) * (y2 - y1) / (x2 - x1)
        return losses[0]

    def update_cable_loss(self, cable):
        print(f"Debug: Updating cable_loss for cable {cable}")
        if cable != "manual":
            try:
                baud_rate = float(self.baud_rate_input.text())
                signaling = self.signaling_input.currentText()
                freq = baud_rate / 2 if signaling == "PAM2" else baud_rate / 4
                print(f"Debug: Baud rate = {baud_rate}, Signaling = {signaling}, Freq = {freq}")
                loss = self.get_cable_loss_from_db(cable, freq)
                print(f"Debug: Calculated cable loss = {loss}")
                self.cable_loss_per_meter_input.setText(f"-{loss:.2f}")
                self.cable_loss_per_meter_input.setReadOnly(True)
            except ValueError as e:
                print(f"Debug: ValueError in update_cable_loss: {e}")
                self.cable_loss_per_meter_input.setText("-1.00")
        else:
            self.cable_loss_per_meter_input.setReadOnly(False)
            print("Debug: Set to manual mode for cable")

    def update_pcb_card_loss(self, material):
        print(f"Debug: Updating pcb_card_loss for material {material}")
        if material != "manual":
            try:
                baud_rate = float(self.baud_rate_input.text())
                signaling = self.signaling_input.currentText()
                freq = baud_rate / 2 if signaling == "PAM2" else baud_rate / 4
                print(f"Debug: Baud rate = {baud_rate}, Signaling = {signaling}, Freq = {freq}")
                loss = self.get_loss_from_db(material, freq)
                print(f"Debug: Calculated loss = {loss}")
                self.pcb_card_loss_per_inch_input.setText(f"-{loss:.2f}")
                self.pcb_card_loss_per_inch_input.setReadOnly(True)
            except ValueError:
                print(f"Debug: ValueError in update_pcb_card_loss: {e}")
                self.pcb_card_loss_per_inch_input.setText("-1.00")
        else:
            self.pcb_card_loss_per_inch_input.setReadOnly(False)
            print("Debug: Set to manual mode for pcb_card")

    def update_pcb_main_loss(self, material):
        print(f"Debug: Updating pcb_main_loss for material {material}")
        if material != "manual":
            try:
                baud_rate = float(self.baud_rate_input.text())
                signaling = self.signaling_input.currentText()
                freq = baud_rate / 2 if signaling == "PAM2" else baud_rate / 4
                print(f"Debug: Baud rate = {baud_rate}, Signaling = {signaling}, Freq = {freq}")
                loss = self.get_loss_from_db(material, freq)
                print(f"Debug: Calculated loss = {loss}")
                self.pcb_main_loss_per_inch_input.setText(f"-{loss:.2f}")
                self.pcb_main_loss_per_inch_input.setReadOnly(True)
            except ValueError:
                print(f"Debug: ValueError in update_pcb_main_loss: {e}")
                self.pcb_main_loss_per_inch_input.setText("-1.00")
        else:
            self.pcb_main_loss_per_inch_input.setReadOnly(False)
            print("Debug: Set to manual mode for pcb_main")
    
    def update_pcb_main2_loss(self, material):
        print(f"Debug: Updating pcb_main2_loss for material {material}")
        if material != "manual":
            try:
                baud_rate = float(self.baud_rate_input.text())
                signaling = self.signaling_input.currentText()
                freq = baud_rate / 2 if signaling == "PAM2" else baud_rate / 4
                print(f"Debug: Baud rate = {baud_rate}, Signaling = {signaling}, Freq = {freq}")
                loss = self.get_loss_from_db(material, freq)
                print(f"Debug: Calculated loss = {loss}")
                self.pcb_main2_loss_per_inch_input.setText(f"-{loss:.2f}")
                self.pcb_main2_loss_per_inch_input.setReadOnly(True)
            except ValueError as e:
                print(f"Debug: ValueError in update_pcb_main2_loss: {e}")
                self.pcb_main2_loss_per_inch_input.setText("-1.00")
        else:
            self.pcb_main2_loss_per_inch_input.setReadOnly(False)
            print("Debug: Set to manual mode for pcb_main2")

    def update_sw_pcb_card_loss(self, material):
        print(f"Debug: Updating sw_pcb_card_loss for material {material}")
        if material != "manual":
            try:
                baud_rate = float(self.baud_rate_input.text())
                signaling = self.signaling_input.currentText()
                freq = baud_rate / 2 if signaling == "PAM2" else baud_rate / 4
                print(f"Debug: Baud rate = {baud_rate}, Signaling = {signaling}, Freq = {freq}")
                loss = self.get_loss_from_db(material, freq)
                print(f"Debug: Calculated loss = {loss}")
                self.sw_pcb_card_loss_per_inch_input.setText(f"-{loss:.2f}")
                self.sw_pcb_card_loss_per_inch_input.setReadOnly(True)
            except ValueError as e:
                print(f"Debug: ValueError in update_sw_pcb_card_loss: {e}")
                self.sw_pcb_card_loss_per_inch_input.setText("-1.00")
        else:
            self.sw_pcb_card_loss_per_inch_input.setReadOnly(False)
            print("Debug: Set to manual mode for sw_pcb_card")

    def update_sw_pcb_main_loss(self, material):
        print(f"Debug: Updating sw_pcb_main_loss for material {material}")
        if material != "manual":
            try:
                baud_rate = float(self.baud_rate_input.text())
                signaling = self.signaling_input.currentText()
                freq = baud_rate / 2 if signaling == "PAM2" else baud_rate / 4
                print(f"Debug: Baud rate = {baud_rate}, Signaling = {signaling}, Freq = {freq}")
                loss = self.get_loss_from_db(material, freq)
                print(f"Debug: Calculated loss = {loss}")
                self.sw_pcb_main_loss_per_inch_input.setText(f"-{loss:.2f}")
                self.sw_pcb_main_loss_per_inch_input.setReadOnly(True)
            except ValueError as e:
                print(f"Debug: ValueError in update_sw_pcb_main_loss: {e}")
                self.sw_pcb_main_loss_per_inch_input.setText("-1.00")
        else:
            self.sw_pcb_main_loss_per_inch_input.setReadOnly(False)
            print("Debug: Set to manual mode for sw_pcb_main")

    def update_sw_pcb_main2_loss(self, material):
        print(f"Debug: Updating sw_pcb_main2_loss for material {material}")
        if material != "manual":
            try:
                baud_rate = float(self.baud_rate_input.text())
                signaling = self.signaling_input.currentText()
                freq = baud_rate / 2 if signaling == "PAM2" else baud_rate / 4
                print(f"Debug: Baud rate = {baud_rate}, Signaling = {signaling}, Freq = {freq}")
                loss = self.get_loss_from_db(material, freq)
                print(f"Debug: Calculated loss = {loss}")
                self.sw_pcb_main2_loss_per_inch_input.setText(f"-{loss:.2f}")
                self.sw_pcb_main2_loss_per_inch_input.setReadOnly(True)
            except ValueError as e:
                print(f"Debug: ValueError in update_sw_pcb_main2_loss: {e}")
                self.sw_pcb_main2_loss_per_inch_input.setText("-1.00")
        else:
            self.sw_pcb_main2_loss_per_inch_input.setReadOnly(False)
            print("Debug: Set to manual mode for sw_pcb_main2")

    def update_all_material_losses(self):
        print("Debug: update_all_material_losses triggered")
        self.update_pcb_card_loss(self.pcb_card_material_combo.currentText())
        self.update_pcb_main_loss(self.pcb_main_material_combo.currentText())
        self.update_pcb_main2_loss(self.pcb_main2_material_combo.currentText())
        self.update_sw_pcb_card_loss(self.sw_pcb_card_material_combo.currentText())
        self.update_sw_pcb_main_loss(self.sw_pcb_main_material_combo.currentText())
        self.update_sw_pcb_main2_loss(self.sw_pcb_main2_material_combo.currentText())
        self.update_cable_loss(self.cable_material_combo.currentText())
    
    def update_spec_settings(self, spec):
        print(f"Debug: Spec changed to {spec}")
        spec_settings = {
            "No spec": {"baud_rate": None, "signaling": "PAM2"},
            "PCI-Gen6": {"baud_rate": "64.0", "signaling": "PAM4"},
            "PCI-Gen5": {"baud_rate": "32.0", "signaling": "PAM2"},
            "PCI-Gen4": {"baud_rate": "16.0", "signaling": "PAM2"},
            "PCI-Gen3": {"baud_rate": "6.0", "signaling": "PAM2"},
            "50GBASE": {"baud_rate": "53.125", "signaling": "PAM4"},
            "100GBASE": {"baud_rate": "106.25", "signaling": "PAM4"},
            "200GBASE": {"baud_rate": "212.5", "signaling": "PAM4"}
        }
        
        settings = spec_settings.get(spec, spec_settings["No spec"])
        
        # Update baud rate
        if settings["baud_rate"] is not None:
            self.baud_rate_input.setText(settings["baud_rate"])
        # "No spec" leaves baud rate unchanged, so do nothing in that case
        
        # Update signaling
        self.signaling_input.setCurrentText(settings["signaling"])
        
        # Update all losses (triggered by baud rate change if applicable)
        self.update_all_material_losses()


    def calculate_budget(self):
        try:
            ip_loss = float(self.ip_input.text())
            package_loss_per_mm = float(self.package_loss_per_mm_input.text())
            package_length = float(self.package_length_input.text())
            package_loss = package_loss_per_mm * package_length
            via_breakout_loss = float(self.via_breakout_input.text())
            via_in_loss = float(self.via_in_input.text()) if self.b2b_en.isChecked() else 0.0
            via_out_loss = float(self.via_out_input.text()) if self.b2b_en.isChecked() else 0.0
            via_rtin_loss = float(self.via_rtin_input.text()) if self.rt_en.isChecked() else 0.0
            via_rtout_loss = float(self.via_rtout_input.text()) if self.rt_en.isChecked() else 0.0
            pcb_card_loss_per_inch = float(self.pcb_card_loss_per_inch_input.text())
            pcb_card_length = float(self.pcb_card_length_input.text())
            pcb_card_loss = pcb_card_loss_per_inch * pcb_card_length
            b2b_con_loss = float(self.b2b_con_input.text()) if self.b2b_en.isChecked() else 0.0
            pcb_main_loss_per_inch = float(self.pcb_main_loss_per_inch_input.text())
            pcb_main_length = float(self.pcb_main_length_input.text())
            pcb_main_loss = pcb_main_loss_per_inch * pcb_main_length
            rt_loss = float(self.rt_input.text()) if self.rt_en.isChecked() else 0.0
            pcb_main2_loss_per_inch = float(self.pcb_main2_loss_per_inch_input.text())
            pcb_main2_length = float(self.pcb_main2_length_input.text())
            pcb_main2_loss = pcb_main2_loss_per_inch * pcb_main2_length
            connector_loss = float(self.connector_input.text()) if self.con_en.isChecked() else 0.0
            cable_loss_per_meter = float(self.cable_loss_per_meter_input.text()) if self.con_en.isChecked() else 0.0
            cable_length = float(self.cable_length_input.text())
            cable_loss = cable_loss_per_meter * cable_length

            sw_ip_loss = float(self.sw_ip_input.text())
            sw_package_loss_per_mm = float(self.sw_package_loss_per_mm_input.text())
            sw_package_length = float(self.sw_package_length_input.text())
            sw_package_loss = sw_package_loss_per_mm * sw_package_length
            via_sw_b_loss = float(self.sw_via_breakout_input.text())
            via_sw_in_loss = float(self.sw_via_in_input.text()) if self.sw_b2b_en.isChecked() else 0.0
            via_sw_out_loss = float(self.sw_via_out_input.text()) if self.sw_b2b_en.isChecked() else 0.0
            via_sw_rtin_loss = float(self.sw_via_rtin_input.text()) if self.sw_rt_en.isChecked() else 0.0
            via_sw_rtout_loss = float(self.sw_via_rtout_input.text()) if self.sw_rt_en.isChecked() else 0.0
            sw_pcb_card_loss_per_inch = float(self.sw_pcb_card_loss_per_inch_input.text())
            sw_pcb_card_length = float(self.sw_pcb_card_length_input.text())
            sw_pcb_card_loss = sw_pcb_card_loss_per_inch * sw_pcb_card_length
            sw_b2b_con_loss = float(self.sw_b2b_con_input.text()) if self.sw_b2b_en.isChecked() else 0.0
            sw_pcb_main_loss_per_inch = float(self.sw_pcb_main_loss_per_inch_input.text())
            sw_pcb_main_length = float(self.sw_pcb_main_length_input.text())
            sw_pcb_main_loss = sw_pcb_main_loss_per_inch * sw_pcb_main_length
            rt_sw_loss = float(self.sw_rt_input.text()) if self.sw_rt_en.isChecked() else 0.0
            sw_pcb_main2_loss_per_inch = float(self.sw_pcb_main2_loss_per_inch_input.text())
            sw_pcb_main2_length = float(self.sw_pcb_main2_length_input.text())
            sw_pcb_main2_loss = sw_pcb_main2_loss_per_inch * sw_pcb_main2_length
            connector_sw_loss = float(self.sw_connector_input.text()) if self.con_en.isChecked() else 0.0
            
            total_loss = (ip_loss + package_loss + via_breakout_loss + via_in_loss + via_out_loss +
                          via_rtin_loss + via_rtout_loss + pcb_card_loss + b2b_con_loss +
                          pcb_main_loss + rt_loss + pcb_main2_loss + connector_loss + cable_loss +
                          sw_ip_loss + sw_package_loss + via_sw_b_loss + via_sw_in_loss + via_sw_out_loss +
                          via_sw_rtin_loss + via_sw_rtout_loss + sw_pcb_card_loss + sw_b2b_con_loss +
                          sw_pcb_main_loss + rt_sw_loss + sw_pcb_main2_loss + connector_sw_loss )
            self.result_label.setText(f"Total Loss: {total_loss:.2f} dB")

            # Determine spec and category from spec_combo
            spec = self.spec_combo.currentText()
            print(f"Debug: Selected spec = {spec}")
            rt_enabled = self.rt_en.isChecked()
            sw_rt_enabled = self.sw_rt_en.isChecked()
            # Only process PCI specs for now
            if "PCI-Gen" in spec and spec != "No spec":
                category = "PCI"
                limit = self.insertion_loss_table.get(category, {}).get(spec)
                print(f"Debug: Category = {category}, Insertion Loss Limit = {limit}")
                # Define segment losses based on RT and sw_RT enable states
                segments = []
                

                if rt_enabled and sw_rt_enabled:
                    # Three segments: IP to RT, RT to sw_RT, sw_RT to IP_switch
                    ip_to_rt = (ip_loss + package_loss + via_breakout_loss + via_in_loss + via_out_loss +
                                pcb_card_loss + b2b_con_loss + pcb_main_loss + via_rtin_loss)
                    rt_to_sw_rt = (rt_loss + via_rtout_loss + pcb_main2_loss + connector_loss + cable_loss +
                                connector_sw_loss + sw_pcb_main2_loss + via_sw_rtin_loss)
                    sw_rt_to_ip_switch = (rt_sw_loss + via_sw_rtout_loss + sw_pcb_main_loss + sw_b2b_con_loss +
                                        sw_pcb_card_loss + via_sw_in_loss + via_sw_out_loss + via_sw_b_loss +
                                        sw_package_loss + sw_ip_loss)
                    segments = [("IP to RT", ip_to_rt), ("RT to sw_RT", rt_to_sw_rt), ("sw_RT to IP_switch", sw_rt_to_ip_switch)]
                elif rt_enabled and not sw_rt_enabled:
                    # Two segments: IP to RT, RT to IP_switch
                    ip_to_rt = (ip_loss + package_loss + via_breakout_loss + via_in_loss + via_out_loss +
                                pcb_card_loss + b2b_con_loss + pcb_main_loss + via_rtin_loss)
                    rt_to_ip_switch = (rt_loss + via_rtout_loss + pcb_main2_loss + connector_loss + cable_loss +
                                    connector_sw_loss + sw_pcb_main2_loss + sw_pcb_main_loss + sw_b2b_con_loss +
                                    sw_pcb_card_loss + via_sw_in_loss + via_sw_out_loss + via_sw_b_loss +
                                    sw_package_loss + sw_ip_loss)
                    segments = [("IP to RT", ip_to_rt), ("RT to IP_sw", rt_to_ip_switch)]
                elif not rt_enabled and sw_rt_enabled:
                    # Two segments: IP to sw_RT, sw_RT to IP_switch
                    ip_to_sw_rt = (ip_loss + package_loss + via_breakout_loss + via_in_loss + via_out_loss +
                                pcb_card_loss + b2b_con_loss + pcb_main_loss + pcb_main2_loss +
                                connector_loss + cable_loss + connector_sw_loss + sw_pcb_main2_loss +
                                via_sw_rtin_loss)
                    sw_rt_to_ip_switch = (rt_sw_loss + via_sw_rtout_loss + sw_pcb_main_loss + sw_b2b_con_loss +
                                        sw_pcb_card_loss + via_sw_in_loss + via_sw_out_loss + via_sw_b_loss +
                                        sw_package_loss + sw_ip_loss)
                    segments = [("IP to sw_RT", ip_to_sw_rt), ("sw_RT to IP_sw", sw_rt_to_ip_switch)]
                else:  # Neither RT nor sw_RT enabled
                    # One segment: IP to IP_switch
                    ip_to_ip_switch = total_loss  # Everything from IP to IP_switch
                    segments = [("IP to IP_sw", ip_to_ip_switch)]

                
                # Build styled and aligned result string with HTML table
                result_text = f"<div style='font-size: 14px;'>Total Loss: {total_loss:.2f} dB</div><br>"
                result_text += "<table style='font-size: 14px;'>"
                for segment_name, segment_loss in segments:
                    status = "Pass" if abs(segment_loss) <= limit else "Fail"
                    status_style = ("background-color: green; padding: 2px;" if status == "Pass" else
                                    "background-color: red; color: yellow; padding: 2px;")
                    result_text += (f"<tr><td>{segment_name}:</td><td>{segment_loss:.2f} dB "
                                f"(<span style='{status_style}'>{status}</span>, Limit: {limit} dB)</td></tr>")
                    print(f"Debug: {segment_name}: {segment_loss:.2f} dB, Status: {status}")
                result_text += "</table>"
                self.result_label.setText(result_text)
            elif spec in ["50GBASE", "100GBASE", "200GBASE"]:
                # IEEE specs
                kr1_limit = self.insertion_loss_table["KR1"].get(spec)
                cr1_limit = self.insertion_loss_table["CR1"].get(spec)
                print(f"Debug: KR1 Limit = {kr1_limit}, CR1 Limit = {cr1_limit}")

                segments = []
                if rt_enabled and sw_rt_enabled:
                    # RT enabled, sw_RT enabled
                    ip_to_rt = (ip_loss + package_loss + via_breakout_loss + via_in_loss + via_out_loss +
                                pcb_card_loss + b2b_con_loss + pcb_main_loss + via_rtin_loss)
                    rt_to_connector = (rt_loss + via_rtout_loss + pcb_main2_loss + connector_loss)
                    ip_sw_to_sw_rt = (sw_ip_loss + sw_package_loss + via_sw_b_loss + via_sw_in_loss + via_sw_out_loss +
                                    sw_pcb_card_loss + sw_b2b_con_loss + sw_pcb_main_loss + via_sw_rtin_loss)
                    sw_rt_to_connector_sw = (rt_sw_loss + via_sw_rtout_loss + sw_pcb_main2_loss + connector_sw_loss)
                    segments = [("IP to RT", ip_to_rt, "KR1", kr1_limit), ("RT to Connector", rt_to_connector, "CR1", cr1_limit),
                                ("IP_sw to sw_RT", ip_sw_to_sw_rt, "KR1", kr1_limit), ("sw_RT to Connector_sw", sw_rt_to_connector_sw, "CR1", cr1_limit)]
                elif rt_enabled and not sw_rt_enabled:
                    # RT enabled, sw_RT disabled
                    ip_to_rt = (ip_loss + package_loss + via_breakout_loss + via_in_loss + via_out_loss +
                                pcb_card_loss + b2b_con_loss + pcb_main_loss + via_rtin_loss)
                    rt_to_connector = (rt_loss + via_rtout_loss + pcb_main2_loss + connector_loss)
                    rt_to_sw_ip = (cable_loss + connector_sw_loss + sw_pcb_main2_loss + sw_pcb_main_loss +
                                sw_b2b_con_loss + sw_pcb_card_loss + via_sw_in_loss + via_sw_out_loss + via_sw_b_loss)
                    sw_ip_to_connector_sw = (sw_package_loss + sw_ip_loss)
                    segments = [("IP to RT", ip_to_rt, "KR1", kr1_limit), ("RT to Connector", rt_to_connector, "CR1", cr1_limit),
                                ("RT to sw_IP", rt_to_sw_ip, "KR1", kr1_limit), ("sw_IP to Connector_sw", sw_ip_to_connector_sw, "CR1", cr1_limit)]
                elif not rt_enabled and sw_rt_enabled:
                    # RT disabled, sw_RT enabled
                    ip_to_connector = (ip_loss + package_loss + via_breakout_loss + via_in_loss + via_out_loss +
                                    pcb_card_loss + b2b_con_loss + pcb_main_loss + pcb_main2_loss + connector_loss)
                    sw_ip_to_sw_rt = (sw_ip_loss + sw_package_loss + via_sw_b_loss + via_sw_in_loss + via_sw_out_loss +
                                    sw_pcb_card_loss + sw_b2b_con_loss + sw_pcb_main_loss + via_sw_rtin_loss)
                    sw_rt_to_connector_sw = (rt_sw_loss + via_sw_rtout_loss + sw_pcb_main2_loss + connector_sw_loss)
                    sw_rt_to_ip = (cable_loss + connector_loss + pcb_main2_loss + pcb_main_loss + b2b_con_loss +
                                pcb_card_loss + via_out_loss + via_in_loss + via_breakout_loss + package_loss + ip_loss)
                    segments = [("IP to Connector", ip_to_connector, "CR1", cr1_limit),
                                ("sw_IP to sw_RT", sw_ip_to_sw_rt, "KR1", kr1_limit), ("sw_RT to Connector_sw", sw_rt_to_connector_sw, "CR1", cr1_limit),
                                ("sw_RT to IP", sw_rt_to_ip, "KR1", kr1_limit)]
                else:
                    # RT disabled, sw_RT disabled
                    ip_to_sw_ip = (ip_loss + package_loss + via_breakout_loss + via_in_loss + via_out_loss +
                                pcb_card_loss + b2b_con_loss + pcb_main_loss + pcb_main2_loss +
                                connector_loss + cable_loss + connector_sw_loss + sw_pcb_main2_loss +
                                sw_pcb_main_loss + sw_b2b_con_loss + sw_pcb_card_loss + via_sw_in_loss +
                                via_sw_out_loss + via_sw_b_loss)
                    segments = [("IP to sw_IP", ip_to_sw_ip, "KR1", kr1_limit)]

                result_text = f"<div style='font-size: 14px;'>Total Loss: {total_loss:.2f} dB</div><br>"
                result_text += "<table style='font-size: 14px;'>"
                for segment_name, segment_loss, limit_type, limit in segments:
                    status = "Pass" if abs(segment_loss) <= limit else "Fail"
                    status_style = ("background-color: green; padding: 2px;" if status == "Pass" else
                                    "background-color: red; color: yellow; padding: 2px;")
                    result_text += (f"<tr><td>{segment_name}:</td><td>{segment_loss:.2f} dB "
                                f"(<span style='{status_style}'>{status}</span>, {limit_type} Limit: {limit} dB)</td></tr>")
                    print(f"Debug: {segment_name}: {segment_loss:.2f} dB, {limit_type} Limit: {limit}, Status: {status}")
                result_text += "</table>"
                self.result_label.setText(result_text)

            else:
                # "No spec"
                self.result_label.setText(f"<div style='font-size: 14px;'>Total Loss: {total_loss:.2f} dB</div>")
            
            self.draw_cascade(ip_loss, package_loss, via_breakout_loss, via_in_loss, via_out_loss,
                              via_rtin_loss, via_rtout_loss, pcb_card_loss, b2b_con_loss,
                              pcb_main_loss, rt_loss, pcb_main2_loss, connector_loss, cable_loss,
                              sw_ip_loss, sw_package_loss, via_sw_b_loss, via_sw_in_loss, via_sw_out_loss,
                              via_sw_rtin_loss, via_sw_rtout_loss, sw_pcb_card_loss, sw_b2b_con_loss,
                              sw_pcb_main_loss, rt_sw_loss, sw_pcb_main2_loss, connector_sw_loss)
        except ValueError:
            self.result_label.setText("Error: Enter valid numbers")

    def draw_via(self, via_center_x, y, pen, gray_brush):
        self.scene.addEllipse(via_center_x, y, self.via_ellipse_size, self.via_ellipse_size, pen)
        self.scene.addEllipse(via_center_x, y + self.block_height - self.via_ellipse_size, self.via_ellipse_size, self.via_ellipse_size, pen)
        self.scene.addEllipse(via_center_x + self.delta_via, y + 7.5, self.via_small_ellipse_size, self.via_small_ellipse_size, pen, gray_brush)
        self.scene.addEllipse(via_center_x + self.delta_via, y + self.block_height - 12.5, self.via_small_ellipse_size, self.via_small_ellipse_size, pen, gray_brush)
        self.scene.addLine(via_center_x + self.delta_via, y + 10, via_center_x + self.delta_via, y + self.block_height - 10, pen)
        self.scene.addLine(via_center_x + self.delta_via + self.via_small_ellipse_size, y + 10, via_center_x + self.delta_via + self.via_small_ellipse_size, y + self.block_height - 10, pen)

    def draw_pcb(self, x, y, pen, col1, col2):
        self.scene.addRect(x + 10, y, self.block_width - 20, self.block_height, pen, col1)
        self.scene.addEllipse(x, y, 20, self.block_height, pen, col1)
        self.scene.addEllipse(x + self.block_width - 20, y, 20, self.block_height, pen, col2)

    def redraw_cascade(self):
        self.calculate_budget()

    def draw_cascade(self, ip_loss, package_loss, via_breakout_loss, via_in_loss, via_out_loss,
                     via_rtin_loss, via_rtout_loss, pcb_card_loss, b2b_con_loss,
                     pcb_main_loss, rt_loss, pcb_main2_loss, connector_loss, cable_loss,
                     sw_ip_loss, sw_package_loss, via_sw_b_loss, via_sw_in_loss, via_sw_out_loss,
                     via_sw_rtin_loss, via_sw_rtout_loss, sw_pcb_card_loss, sw_b2b_con_loss,
                     sw_pcb_main_loss, rt_sw_loss, sw_pcb_main2_loss, connector_sw_loss):
        self.scene.clear()
        pen = QPen(QColor("black"), 2)
        gray_brush = QBrush(QColor("gray"))
        dark_gray_brush = QBrush(QColor("#555555"))
        silver_brush = QBrush(QColor("#C0C0C0"))
        light_green_brush = QBrush(QColor("#90EE90"))
        olive_brush = QBrush(QColor("#808000"))

        y_top = -50
        y_bottom = y_top + 4 * self.block_height
        spacing = 32
        close_spacing = 10
        small_square_size = 30
        x_init = -100

        b2b_enabled = self.b2b_en.isChecked()
        sw_b2b_enabled = self.sw_b2b_en.isChecked()
        rt_enabled = self.rt_en.isChecked()
        sw_rt_enabled = self.sw_rt_en.isChecked()
        con_enabled = self.con_en.isChecked()

        x = x_init
        self.scene.addRect(x, y_top + (self.block_height - small_square_size) // 2, small_square_size, small_square_size, pen, QBrush(QColor("black")))
        self.scene.addText(f"IP\n{ip_loss:.1f}").setPos(x + 10, y_top - self.block_height * 1.5)
        self.scene.addRect(x, y_bottom + (self.block_height - small_square_size) // 2, small_square_size, small_square_size, pen, QBrush(QColor("black")))
        self.scene.addText(f"IP\n{sw_ip_loss:.1f}").setPos(x + 10, y_bottom - self.block_height * 1.5)

        x += small_square_size + spacing
        package_x = x
        self.scene.addRect(x, y_top, self.block_width, self.block_height, pen, silver_brush)
        self.scene.addText(f"Package\n{package_loss:.1f}").setPos(x + 10, y_top - self.block_height * 1.5)
        self.scene.addRect(x, y_bottom, self.block_width, self.block_height, pen, silver_brush)
        self.scene.addText(f"Package\n{sw_package_loss:.1f}").setPos(x + 10, y_bottom - self.block_height * 1.5)

        x += self.block_width + close_spacing
        via_breakout_x = x + self.via_ellipse_size // 2
        via_center_x = x
        self.draw_via(via_center_x, y_top, pen, gray_brush)
        self.scene.addText(f"VIA\nbreakout\n{via_breakout_loss:.1f}").setPos(x, y_top - self.block_height * 1.5)
        self.draw_via(via_center_x, y_bottom, pen, gray_brush)
        self.scene.addText(f"VIA sw\nbreakout\n{via_sw_b_loss:.1f}").setPos(x, y_bottom - self.block_height * 1.5)

        x += self.via_ellipse_size + spacing
        pcb_card_x = x
        self.draw_pcb(x, y_top, pen, olive_brush, light_green_brush)
        self.scene.addText(f"PCB_card\n{pcb_card_loss:.1f}").setPos(x + 10, y_top - self.block_height * 1.5)
        self.draw_pcb(x, y_bottom, pen, olive_brush, light_green_brush)
        self.scene.addText(f"PCB_card\n{sw_pcb_card_loss:.1f}").setPos(x + 10, y_bottom - self.block_height * 1.5)

        if b2b_enabled or sw_b2b_enabled:
            x += self.block_width + spacing
            via_in_x = x + self.via_ellipse_size // 2
            via_center_x = x
            if b2b_enabled:
                self.draw_via(via_center_x, y_top, pen, gray_brush)
                self.scene.addText(f"VIA\nin\n{via_in_loss:.1f}").setPos(x, y_top - self.block_height * 1.5)
            if sw_b2b_enabled:
                self.draw_via(via_center_x, y_bottom, pen, gray_brush)
                self.scene.addText(f"VIA\nsw_in\n{via_sw_in_loss:.1f}").setPos(x, y_bottom - self.block_height * 1.5)

            x += self.via_ellipse_size + close_spacing
            b2b_con_x = x
            gradient = QLinearGradient(x, y_top, x + self.block_width, y_top)
            gradient.setColorAt(0, QColor("Black"))
            gradient.setColorAt(0.25, QColor("Gray"))
            gradient.setColorAt(0.5, QColor("White"))
            gradient.setColorAt(0.75, QColor("Gray"))
            gradient.setColorAt(1, QColor("Black"))
            b2b_con_brush = QBrush(gradient)
            if b2b_enabled:
                self.scene.addRect(x, y_top, self.block_width, self.block_height, pen, b2b_con_brush)
                self.scene.addText(f"B2B_con\n{b2b_con_loss:.1f}").setPos(x + 10, y_top - self.block_height * 1.5)
            if sw_b2b_enabled:
                self.scene.addRect(x, y_bottom, self.block_width, self.block_height, pen, b2b_con_brush)
                self.scene.addText(f"B2B_con\n{sw_b2b_con_loss:.1f}").setPos(x + 10, y_bottom - self.block_height * 1.5)

            x += self.block_width + close_spacing
            via_out_x = x + self.via_ellipse_size // 2
            via_center_x = x
            if b2b_enabled:
                self.draw_via(via_center_x, y_top, pen, gray_brush)
                self.scene.addText(f"VIA\nout\n{via_out_loss:.1f}").setPos(x, y_top - self.block_height * 1.5)
            if sw_b2b_enabled:
                self.draw_via(via_center_x, y_bottom, pen, gray_brush)
                self.scene.addText(f"VIA\nsw_out\n{via_sw_out_loss:.1f}").setPos(x, y_bottom - self.block_height * 1.5)
        else:
            via_in_x = pcb_card_x + self.block_width
            b2b_con_x = via_in_x
            via_out_x = b2b_con_x
            x = via_out_x

        x += self.via_ellipse_size + spacing if b2b_enabled or sw_b2b_enabled else spacing
        pcb_main_x = x
        self.draw_pcb(x, y_top, pen, olive_brush, light_green_brush)
        self.scene.addText(f"PCB\nmain\n{pcb_main_loss:.1f}").setPos(x + 10, y_top - self.block_height * 1.5)
        self.draw_pcb(x, y_bottom, pen, olive_brush, light_green_brush)
        self.scene.addText(f"PCB\nsw_main\n{sw_pcb_main_loss:.1f}").setPos(x + 10, y_bottom - self.block_height * 1.5)

        if rt_enabled or sw_rt_enabled:
            x += self.block_width + spacing
            via_rtin_x = x + self.via_ellipse_size // 2
            via_center_x = x
            if rt_enabled:
                self.draw_via(via_center_x, y_top, pen, gray_brush)
                self.scene.addText(f"VIA\nRTin\n{via_rtin_loss:.1f}").setPos(x, y_top - self.block_height * 1.5)
            if sw_rt_enabled:
                self.draw_via(via_center_x, y_bottom, pen, gray_brush)
                self.scene.addText(f"VIA sw\nRTin\n{via_sw_rtin_loss:.1f}").setPos(x, y_bottom - self.block_height * 1.5)

            x += self.via_ellipse_size + close_spacing
            rt_x = x
            if rt_enabled:
                self.scene.addRect(x, y_top, small_square_size, small_square_size, pen, dark_gray_brush)
                self.scene.addText(f"RT\n{rt_loss:.1f}").setPos(x + 10, y_top - self.block_height * 1.5)
            if sw_rt_enabled:
                self.scene.addRect(x, y_bottom, small_square_size, small_square_size, pen, dark_gray_brush)
                self.scene.addText(f"RT\nsw\n{rt_sw_loss:.1f}").setPos(x + 10, y_bottom - self.block_height * 1.5)

            x += small_square_size + close_spacing
            via_rtout_x = x + self.via_ellipse_size // 2
            via_center_x = x
            if rt_enabled:
                self.draw_via(via_center_x, y_top, pen, gray_brush)
                self.scene.addText(f"VIA\nRTout\n{via_rtout_loss:.1f}").setPos(x, y_top - self.block_height * 1.5)
            if sw_rt_enabled:
                self.draw_via(via_center_x, y_bottom, pen, gray_brush)
                self.scene.addText(f"VIA sw\nRTout\n{via_sw_rtout_loss:.1f}").setPos(x, y_bottom - self.block_height * 1.5)
        else:
            via_rtin_x = pcb_main_x + self.block_width
            rt_x = via_rtin_x
            via_rtout_x = rt_x
            x = via_rtout_x

        x += self.via_ellipse_size + spacing if rt_enabled or sw_rt_enabled else spacing
        pcb_main2_x = x
        self.draw_pcb(x, y_top, pen, olive_brush, light_green_brush)
        self.scene.addText(f"PCB\nmain2\n{pcb_main2_loss:.1f}").setPos(x + 10, y_top - self.block_height * 1.5)
        self.draw_pcb(x, y_bottom, pen, olive_brush, light_green_brush)
        self.scene.addText(f"PCB\nswitch2\n{sw_pcb_main2_loss:.1f}").setPos(x + 10, y_bottom - self.block_height * 1.5)

        if con_enabled:
            x += self.block_width + spacing
            connector_x = x
            self.scene.addRect(x, y_top, self.block_width, self.block_height, pen)
            self.scene.addText(f"Connector\n{connector_loss:.1f}").setPos(x + 10, y_top - self.block_height * 1.5)
            self.scene.addRect(x, y_bottom, self.block_width, self.block_height, pen)
            self.scene.addText(f"Connector_sw\n{connector_sw_loss:.1f}").setPos(x + 10, y_bottom - self.block_height * 1.5)

            cable_start_x = connector_x + self.block_width
            cable_y_top = y_top + self.block_height // 2
            cable_y_bottom = y_bottom + self.block_height // 2
            step_length = 20
            mid_y = (cable_y_top + cable_y_bottom) / 2
            cable_pen = QPen(QColor("black"), 10, Qt.PenStyle.SolidLine)
            x1 = cable_start_x
            x2 = x1 + step_length
            y1 = cable_y_top
            self.scene.addLine(x1, y1, x2, y1, cable_pen)
            x3 = x2
            y2 = mid_y
            self.scene.addLine(x2, y1, x3, y2, cable_pen)
            x4 = x3 + step_length
            y3 = y2
            self.scene.addLine(x3, y2, x4, y3, cable_pen)
            x5 = x4
            y4 = cable_y_bottom
            self.scene.addLine(x4, y3, x5, y4, cable_pen)
            x6 = x5 - 2 * step_length
            y5 = y4
            self.scene.addLine(x5, y4, x6, y5, cable_pen)
            label_x = x4 + 5
            label_y = (y3 + y4) / 2 - 10
            self.scene.addText(f"Cable\n{cable_loss:.1f}").setPos(label_x, label_y)

        for y_h in [y_top, y_bottom]:
            self.scene.addLine(x_init + small_square_size, y_h + self.block_height // 2, package_x, y_h + self.block_height // 2, pen)
            self.scene.addLine(package_x + self.block_width, y_h + self.block_height // 2, via_breakout_x, y_h, pen)
            self.scene.addLine(via_breakout_x, y_h + self.block_height, pcb_card_x, y_h + self.block_height // 2, pen)
            if (b2b_enabled and y_h == y_top) or (sw_b2b_enabled and y_h == y_bottom):
                self.scene.addLine(pcb_card_x + self.block_width, y_h + self.block_height // 2, via_in_x, y_h, pen)
                self.scene.addLine(via_in_x, y_h + self.block_height, b2b_con_x, y_h + self.block_height // 2, pen)
                self.scene.addLine(b2b_con_x + self.block_width, y_h + self.block_height // 2, via_out_x, y_h, pen)
                self.scene.addLine(via_out_x, y_h + self.block_height, pcb_main_x, y_h + self.block_height // 2, pen)
            else:
                self.scene.addLine(pcb_card_x + self.block_width, y_h + self.block_height // 2, pcb_main_x, y_h + self.block_height // 2, pen)
            if (rt_enabled and y_h == y_top) or (sw_rt_enabled and y_h == y_bottom):
                self.scene.addLine(pcb_main_x + self.block_width, y_h + self.block_height // 2, via_rtin_x, y_h, pen)
                self.scene.addLine(via_rtin_x, y_h + self.block_height, rt_x, y_h + self.block_height // 2, pen)
                self.scene.addLine(rt_x + small_square_size, y_h + self.block_height // 2, via_rtout_x, y_h, pen)
                self.scene.addLine(via_rtout_x, y_h + self.block_height, pcb_main2_x, y_h + self.block_height // 2, pen)
            else:
                self.scene.addLine(pcb_main_x + self.block_width, y_h + self.block_height // 2, pcb_main2_x, y_h + self.block_height // 2, pen)
            if con_enabled:
                self.scene.addLine(pcb_main2_x + self.block_width, y_h + self.block_height // 2, connector_x, y_h + self.block_height // 2, pen)
            else:
                self.scene.addLine(pcb_main2_x + self.block_width-10, y_top + self.block_height // 2, pcb_main2_x + self.block_width-10, y_bottom + self.block_height // 2, pen)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LinkBudgetWindow()
    window.show()
    sys.exit(app.exec())