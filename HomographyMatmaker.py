import glfw
from OpenGL.GL import *
import numpy as np
import glm as m

roation  = 0


<<<<<<< Updated upstream
def main():
=======

roation  = random.randint(15,1000)
window_size =640
r_scn = 1
screen_ratio=2 #screen pixel ratio
output =cv2.imread('output.png')
final_points = []
points=[]
save =True

def save_image(image, filename):
    image = Image.fromarray(image)
    image.save(filename)
    # extra()

def draw_circle(event, x, y, flags, param):
    print('Helo')
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Mouse clicked at: ({x}, {y})")
        
def main(orig_image_id):
>>>>>>> Stashed changes
    # Initialize GLFW
    if not glfw.init():
        return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # Required on Mac
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE) 

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 640, "Hello Triangle", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)
    print(glGetString(GL_VERSION))
    # Triangle vertices
    vertices = np.array([
        -0.5, -0.5, 0,  # Bottom left
        0.5, -0.5,  0,   # Bottom right
        0.5, 0.5,  0,    # Top right
        -0.5, -0.5,  0,  # Bottom left
        0.5, 0.5,  0,    # Top right
        -0.5, 0.5, 0    # Top left
    ], dtype=np.float32)

    # vertices = np.array([
    #    0.0687989,  0.249017, 0,  # Bottom left
    #      0.17313,-0.594062,  0,   # Bottom right
    #     -0.133474, -0.483108,  0,    # Top right
    #    0.0687989,  0.249017, 0,   # Bottom left
    #     -0.133474,  -0.483108,  0,    # Top right
    #     -0.229139, 0.786211, 0    # Top left
    # ], dtype=np.float32)

    # Vertex Shader
    vertex_shader = """
    #version 400
    in vec3 position;
    uniform mat4 rot;
    void main()
    {
        gl_Position =  rot*vec4(position, 1.0);
    }
    """

    # Fragment Shader
    fragment_shader = """
    #version 400
    out vec4 color;
    void main()
    {
        color = vec4(1.0, 0.5, 0.2, 1.0);
    }
    """

    # Create Shader Program
    shader_program = glCreateProgram()
    glAttachShader(shader_program, compile_shader(GL_VERTEX_SHADER, vertex_shader))
    glAttachShader(shader_program, compile_shader(GL_FRAGMENT_SHADER, fragment_shader))
    glLinkProgram(shader_program)

    # Set up vertex array object (VAO)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # Vertex buffer object (VBO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Vertex attributes
<<<<<<< Updated upstream
    position = glGetAttribLocation(shader_program, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(position)

=======
    # position = glGetAttribLocation(shader_program, 'position')
    # glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)
    # glEnableVertexAttribArray(position)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)
    
    #texture setup
    global points
    image,points =getMarkedIMageandPoints(orig_image_id)
    points = [[x[0],x[1]] for x in points]
    texture_id   = create_texture(image)
>>>>>>> Stashed changes
#     # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Render here
        glClear(GL_COLOR_BUFFER_BIT)

        # Draw triangle
        glUseProgram(shader_program)
        glBindVertexArray(VAO)
        position = glGetAttribLocation(shader_program, 'position')
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(position)
        extra(shader_program)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()
<<<<<<< Updated upstream

    glfw.terminate()
=======
        if count < 100 and save :
            width, height = glfw.get_window_size(window)
            img=capture_image(width,height)
            
            save_image(img, 'Dataset/img/output_' + str(orig_image_id) + "_" + str(count) +'.png')
            with open('Dataset/trasn/output_' + str(orig_image_id) + "_" + str(count) +'.pkl', 'wb') as f:
                pickle.dump(rotmat.to_list(), f)
            # global output
            # output = cv2.imread('output.png')
        else:
            exit()
            
            
        if(count == 20):
            # cv2.destroyAllWindows()
            # glfw.terminate()
            # break
            pass
        # cv2.imshow("destination ",output)
        
        #save image
        
        count +=1
        
        
        
        
    points = points[:len(final_points)]
    homo,_ = cv2.findHomography(640*np.float32(points),640*np.float32(final_points))
    
    print(homo)
    print(points)
    print(final_points)
    w,h =glfw.get_window_size(window)
    image = cv2.resize(image,(w,h))
    final_image =cv2.warpPerspective(image,homo,(w,h))
    cv2.imshow('final image',final_image)
    cv2.waitKey(0)
    
    # global output
    # output = cv2.imread('output.png')
    # reference=cv2.imread('keypoints.jpg')
    # cv2.namedWindow('destination')
    
    
    # while True:
    #     # cv2.imshow("reference ",reference)
    #     cv2.imshow("destination ",output)
    #     cv2.setMouseCallback('destination',draw_circle)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
    #         break
    # exit()
    return
>>>>>>> Stashed changes

def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    
    return shader


def extra(program):
    global roation
    roation = 5
    roation1 =0
    rotMatz =m.rotate(roation/2,m.vec3(0,0,1))
    rotMaty =m.rotate(roation/3,m.vec3(0,1,0))
    rotMatx =m.rotate(roation,m.vec3(1,0,0))

    tranx= m.translate(m.vec3(-(roation%3)/6,0,0))
    trany = m.translate(m.vec3(0,-2*(roation%3)/6,0))
    tranz =  m.translate(m.vec3(0,0,-1.5))

    pers = m.perspective(m.radians(50),1,0.01,40)
    
    test = m.mat4([[-0.5,-0.5,0,1],[ 0.5, -0.5,0,1],[ 0.5, 0.5,0,1],[-0.5, 0.5,0,1]])
    # test=m.transpose(test)


    rotMat =  pers   * tranz * rotMaty
    test2 = rotMat * m.vec4(0.5,0.5,0,1)
    test = rotMat * test
    # global roation
    roation+=0.01
    print('test')
    print(test)
    test =m.transpose(test)
    last_row = test.to_list()[-1]
    # print(last_row)
    matrix_list = test.to_list()
    # for i in range(4):
    matrix_list = [(m.div(m.vec4(x),m.vec4(last_row))).to_list() for x in matrix_list]
    s_last_row = matrix_list[-2]
    matrix_list = [(m.div(m.vec4(x),m.vec4(s_last_row))).to_list() for x in matrix_list]
    # matrix_list[-1] = [1,1,1,1]
    matrix_list = matrix_list[:-1]

    print( '= = = = = =')
# Convert the resulting list of lists back to a PyGLM mat4 object
    result_matrix = m.mat3x4(matrix_list)
    print(m.transpose(result_matrix))
    print(" -------------   " )
    loc =  glGetUniformLocation(program,'rot')
    # rotMat = m.mat4()
    glUniformMatrix4fv(loc,1,GL_FALSE,m.value_ptr(rotMat))


<<<<<<< Updated upstream
=======
def create_texture(image):
    
    img, img_data = load_image(image)
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    return texture_id


def capture_image(width, height):
    glReadBuffer(GL_FRONT)
    height =  screen_ratio*height
    width = screen_ratio*width
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    image = np.flip(image, axis=0)  # Flip the image vertically
    return image

def mouse_button_callback(window, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        global points
        global final_points
        if(len(final_points) >= len(points)):
            return
        w,h=glfw.get_cursor_pos(window)
        W,H =  glfw.get_window_size(window)
        final_points.append([w/W,h/H])
        print(points)
        print(final_points)
        w = screen_ratio *w
        h= screen_ratio *h
        print("Left mouse button pressed "+str(w) + " " +str(h) )
        cv2.circle(output,(int(w),int(h)),6,(0,0,255),-1)
       
    elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        print("Right mouse button pressed")
        

>>>>>>> Stashed changes
    


if __name__ == "__main__":
   
<<<<<<< Updated upstream
print('hello')
roation=0
main()
    # extra()
=======
    print('hello')
    # roation=0
    orig_image_id = 5
    main(orig_image_id)
>>>>>>> Stashed changes
