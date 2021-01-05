from textwrap3 import dedent
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
from src.detect_mask_image import detect_mask
import numpy as np
import base64
import cv2


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.config.suppress_callback_exceptions = True

global face_detector, mask_detector, vid_capture, original_opencv_img, annotated_img, confidence, status, alarm_on
prototxt_path = "./face_detector_model/deploy.prototxt"
weights_path = "./face_detector_model/res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNet(prototxt_path, weights_path)
mask_detector = load_model("./mask_detector_models/mask_detector_MFN.h5")
vid_capture = None
original_opencv_img = None
annotated_img = None
confidence = 50
status = 0
alarm_on = True


# Main App
app.layout = html.Div(
    children=[
        dcc.Interval(id="interval-update", interval=1000, n_intervals=0),
        html.Div(id="top-bar", className="row"),
        html.Div(
            className="container",
            children=[
                html.Div(
                    id="left-side-column",
                    className="eight columns",
                    children=[
                        html.Div(
                            id="header-section",
                            children=[
                                html.H4("Proper Mask Wearing Detection and Alarm System"),
                                html.P(
                                    "To get started, select whether you want to detect an image or webcam feed, "
                                    "and choose the model and the confidence level of your face detector. "
                                    "The results will be marked with bounding boxes in realtime."
                                ),
                            ],
                        ),
                        html.Div(
                            id="video-mode",
                            children=[
                                html.Div(
                                    id="annotated-frame-container",
                                ),
                            ]
                        ),
                        html.Div(
                            id="image-mode",
                            children=[
                                html.Div(
                                    id="annotated-image-container",
                                ),
                                html.Div(
                                    id="upload-div",
                                    children=[dcc.Upload(
                                        id='upload-image',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select Image (.jpg/.jpeg/.png)')
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                        # Allow only one file to be uploaded
                                        multiple=False
                                    )],
                                    style={"visibility": "visible"}
                                ),
                            ],
                        ),
                        html.Div(id="hidden-div-1", style={"display": "none"}),
                        html.Div(id="hidden-div-2", style={"display": "none"}),
                        html.Div(id="hidden-div-3", style={"display": "none"}),
                    ],
                ),
                html.Div(
                    id="right-side-column",
                    className="four columns",
                    children=[
                        html.Div(
                            className="markdown-text",
                            children=[
                                dcc.Markdown(
                                    children=dedent(
                                        """
                                        ##### What am I looking at?
            
                                        This app detects human faces and proper mask wearing in images and webcam 
                                        streams. Under the COVID-19 pandemic, the demand for an effective mask 
                                        detection on embedded systems of limited computing capabilities has surged.
                                        Trained on MobileNetV2, the app is computationally efficient to deploy to 
                                        help control the spread of the disease.
                                        
                                        ##### More about this Dash app
            
                                        The MFN Model is capable of detecting 3 scenarios: 
                                        (1) incorrect mask wearing, (2) correct mask wearing and (3) no mask. 
                                        The RMFD model is trained to classify 2 scenarios: 
                                        (1) mask worn and (2) no mask. To learn more about 
                                        the project, please visit the 
                                        [project repository](https://github.com/achen353/Face-Mask-Detector).
                                        """
                                    )
                                )
                            ],
                        ),
                        html.Div(
                            className="control-element",
                            children=[
                                html.Div(children=["Detection Mode:"]),
                                dcc.RadioItems(
                                    id="detection-mode",
                                    options=[
                                        {
                                            "label": "Image",
                                            "value": "image",
                                        },
                                        {
                                            "label": "Webcam Video",
                                            "value": "video",
                                        },
                                    ],
                                    value="image",
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-element",
                            children=[
                                html.Div(
                                    children=["Minimum Face Detector Confidence Threshold:"]
                                ),
                                html.Div(
                                    dcc.Slider(
                                        id="slider-face-detector-minimum-confidence-threshold",
                                        min=20,
                                        max=80,
                                        marks={
                                            i: f"{i}%"
                                            for i in range(20, 81, 10)
                                        },
                                        value=50,
                                        updatemode="drag",
                                    )
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-element",
                            children=[
                                html.Div(children=["Mask Detector Model Selection:"]),
                                dcc.RadioItems(
                                    id="radio-item-mask-detector-selection",
                                    options=[
                                        {
                                            "label": "MFN Model",
                                            "value": "MFN",
                                        },
                                        {
                                            "label": "RMFD Model",
                                            "value": "RMFD",
                                        },
                                    ],
                                    value="MFN",
                                ),
                            ],
                        ),
                        html.Div(
                            id="alarm-control",
                            className="control-element",
                            children=[
                                html.Div(children=["Alarm on/off:"]),
                                dcc.RadioItems(
                                    id="radio-alarm-switch",
                                    options=[
                                        {
                                            "label": "Alarm on",
                                            "value": "on",
                                        },
                                        {
                                            "label": "Alarm off",
                                            "value": "off",
                                        },
                                    ],
                                    value="on",
                                ),
                            ],
                            style={
                                "visibility": "hidden",
                            },

                        ),
                        html.Div(
                            id="video-control",
                            children=[
                                html.Button(
                                    "Start Video",
                                    id="video-start-button",
                                    n_clicks=0,
                                    className="video-start-button",
                                ),
                                html.Div(
                                    id="hidden-gap",
                                    style={
                                        "margin-right": "1%",
                                        "margin-left": "1%",
                                        "visibility": "none",
                                    }
                                ),
                                html.Button(
                                    "Stop Video",
                                    id="video-stop-button",
                                    n_clicks=0,
                                    className="video-stop-button",
                                )
                            ],
                            style={
                                "visibility": "hidden",
                                "display": "flex",
                                "justify-content": "center",
                                "align-items": "center",
                                "margin-top": "3%",
                            },
                        ),
                        html.Div(
                            id="hidden-audio-div",
                            children=[
                                html.Audio(
                                    id="alert-audio",
                                    src=app.get_asset_url("no_mask_US_female.mp3"),
                                    autoPlay=False,
                                    controls=False,
                                    loop=True
                                )
                            ],
                            style={"visibility": "visible"}
                        )
                    ],
                ),
            ],
        ),
    ]
)


def detect_and_create_html_img(img, img_id):
    global status
    status, output_img = detect_mask(img, face_detector, mask_detector, confidence, False)
    output_base64 = cv2.imencode('.jpg', output_img)[1].tobytes()
    output_base64 = base64.b64encode(output_base64).decode('utf-8')
    output_url = "data:image/;base64,{}".format(output_base64)
    output_html_img = html.Img(
        id=img_id,
        className=img_id,
        src=output_url,
        style={
            "max-width": "100%",
            "height": "auto",
        }
    )
    return output_html_img


@app.callback(
    Output("upload-div", "style"),
    Input("detection-mode", "value"),
)
def display_upload(detection_mode):
    if detection_mode == "video":
        return {"visibility": "hidden"}
    return {"visibility": "visible"}


@app.callback(
    Output("video-control", "style"),
    Input("detection-mode", "value"),
)
def display_video_control(detection_mode):
    if detection_mode == "image":
        return {
            "visibility": "hidden",
            "display": "flex",
            "justify-content": "center",
            "align-items": "center",
            "margin-top": "3%",
        }
    return {
        "visibility": "visible",
        "display": "flex",
        "justify-content": "center",
        "align-items": "center",
        "margin-top": "3%",
    }


@app.callback(
    Output("alarm-control", "style"),
    Input("detection-mode", "value"),
)
def display_alarm_control(detection_mode):
    if detection_mode == "image":
        return {
            "visibility": "hidden",
        }
    return {
        "visibility": "visible",
    }


@app.callback(
    [
        Output("annotated-frame-container", "children"),
        Output("annotated-image-container", "children")
    ],
    [
        Input("upload-image", "contents"),
        Input("video-start-button", "n_clicks"),
        Input("video-stop-button", "n_clicks"),
        Input("interval-update", "n_intervals")
    ],
    [
        State("detection-mode", "value"),
        State("radio-alarm-switch", "value")
    ]
)
def update_image_or_frame(contents, start_n, end_n, n_intervals, detection_mode, alarm_state):
    global vid_capture, original_opencv_img, annotated_img, alarm_on
    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if input_id == "interval-update":
            if detection_mode == "video" and vid_capture is not None and vid_capture.isOpened():
                annotated_img = None
                flags, frame = vid_capture.read()
                annotated_frame = detect_and_create_html_img(frame, "annotated-frame")
                return annotated_frame, []
            elif detection_mode == "image" and annotated_img is not None:
                alarm_on = False
                return [], annotated_img
        if input_id == "video-start-button" and detection_mode == "video" \
                and (vid_capture is None or not vid_capture.isOpened()):
            annotated_img = None
            if alarm_state == "on":
                alarm_on = True
            else:
                alarm_on = False
            vid_capture = cv2.VideoCapture(0)
            return [], []
        if input_id == "video-stop-button" and detection_mode == "video" and \
                vid_capture is not None and vid_capture.isOpened():
            annotated_img = None
            alarm_on = False
            vid_capture.release()
            return [], []
        if input_id == "upload-image" and detection_mode == "image":
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                img_arr = np.frombuffer(decoded, dtype=np.uint8)
                img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
                original_opencv_img = img
                annotated_img = detect_and_create_html_img(img.copy(), "annotated-image")
                return [], annotated_img
            except:
                return [], [html.Div(["Please upload file of the following types: jpg, jpeg, png."])]
    return [], []


@app.callback(
    Output("hidden-div-1", "children"),
    Input("radio-item-mask-detector-selection", "value"),
)
def update_mask_detector(model):
    global mask_detector, annotated_img
    model_path = "./mask_detector_models/mask_detector_" + model + ".h5"
    mask_detector = load_model(model_path)
    if original_opencv_img is not None:
        annotated_img = detect_and_create_html_img(original_opencv_img.copy(), "annotated-image")
    return []


@app.callback(
    Output("hidden-div-2", "children"),
    Input("slider-face-detector-minimum-confidence-threshold", "value"),
)
def update_confidence(confidence_value):
    global confidence, annotated_img
    confidence = confidence_value * 0.01
    if original_opencv_img is not None:
        annotated_img = detect_and_create_html_img(original_opencv_img.copy(), "annotated-image")
    return []


@app.callback(
    Output("hidden-div-3", "children"),
    Input("radio-alarm-switch", "value"),
)
def update_alarm(alarm_state):
    global alarm_on
    if alarm_state == "on":
        alarm_on = True
    else:
        alarm_on = False

    return []


@app.callback(
    [
        Output("alert-audio", "src"),
        Output("alert-audio", "autoPlay")
    ],
    Input("interval-update", "n_intervals"),
    [
        State("detection-mode", "value"),
        State("radio-item-mask-detector-selection", "value")
    ]
)
def play_alert_audio(n_intervals, detection_mode, model):
    no_mask_src = app.get_asset_url("no_mask_US_female.mp3")
    mask_incorrect_src = app.get_asset_url("mask_incorrect_US_female.mp3")
    if detection_mode == "video":
        if model == "MFN":
            if status == 2 and alarm_on is True:
                return no_mask_src, True
            elif status == 1 and alarm_on is True:
                return mask_incorrect_src, True
        else:
            if status == 1 and alarm_on is True:
                return no_mask_src, True
    return "", False


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
