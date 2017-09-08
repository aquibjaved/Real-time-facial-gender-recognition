import os,cv2
import time
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

haar_file = 'haarcascade_frontalface_default.xml'
(width, height) = (112, 92)
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

print "Loading model file...."
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]
frame_count=0

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    while True:

        start_time = time.time()
        (_, im) = webcam.read()
        frame_count += 1
        gray = im
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if frame_count % 10 == 0:

                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (height, width))
                    cv2.imwrite("my_frame.jpg",im)
                    image_data = tf.gfile.FastGFile("./my_frame.jpg", 'rb').read()
                    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                    print("--- %s seconds ---" % (time.time() - start_time))

                    for node_id in top_k:
                        human_string = label_lines[node_id]  # List
                        score = predictions[0][node_id]
                        print human_string[:6], score
                        if score > 0.5:
                            cv2.putText(im,(human_string),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,2,(0, 255, 0))
                cv2.imshow('Gender recognition', im)
                key = cv2.waitKey(10)
                if key == 27:
                    break
