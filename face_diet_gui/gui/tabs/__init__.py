# Tab modules: one class per tab.
from face_diet_gui.gui.tabs.tab1_video_processing import VideoProcessingTab
from face_diet_gui.gui.tabs.tab2_face_instance_review import FaceInstanceReviewTab
from face_diet_gui.gui.tabs.tab3_mismatch_resolution import MismatchResolutionTab
from face_diet_gui.gui.tabs.tab4_face_id_clustering import FaceIDAssignmentTab
from face_diet_gui.gui.tabs.tab5_face_id_review import ManualReviewTab

__all__ = [
    "VideoProcessingTab",
    "FaceInstanceReviewTab",
    "MismatchResolutionTab",
    "FaceIDAssignmentTab",
    "ManualReviewTab",
]
