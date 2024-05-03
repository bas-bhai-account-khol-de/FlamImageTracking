import glfw
from OpenGL.GL import *
import numpy as np
import glm as m
from PIL import Image
from Homographysolver import *
import utils 


roation  = random.randint(15,1000)
window_size =640
r_scn = 2

def main():
    # Initialize GLFW
    if not glfw.init():
        return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # Required on Mac
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE) 

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(window_size * r_scn, window_size, "Hello Triangle", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)
    print(glGetString(GL_VERSION))
    # Triangle vertices
    vertices = np.array([
        -0.5, -0.5, 0.0, 0.0, 0.0, #0
         0.5, -0.5, 0.0, 1.0, 0.0, #1
         0.5,  0.5, 0.0, 1.0, 1.0, #2
        -0.5,  0.5, 0.0, 0.0, 1.0  #3
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2,
        2, 3, 0
    ], dtype=np.uint32)

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
    #version 330 core
    layout (location = 0) in vec3 position;
    layout (location = 1) in vec2 texCoord;
    
    out vec2 TexCoord;
    uniform mat4 rot;
    void main()
    {
        gl_Position = rot * vec4(position, 1.0);
        TexCoord = texCoord;
    }
    """
    

    # Fragment Shader
    fragment_shader = """
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D ourTexture;
    void main()
    {
        FragColor = texture(ourTexture, TexCoord);
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
    
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Vertex attributes
    # position = glGetAttribLocation(shader_program, 'position')
    # glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)
    # glEnableVertexAttribArray(position)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)
    
    #texture setup
    image =getMarkedIMageandPoints()
    texture_id   = create_texture(image)
#     # Loop until the user closes the window
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    while not glfw.window_should_close(window):
        # Render here
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw triangle
        glUseProgram(shader_program)
        glBindVertexArray(VAO)
        
        glUniform1i(glGetUniformLocation(shader_program, "ourTexture"), 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        extra(shader_program)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    glfw.terminate()

def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    
    return shader


def extra(program):
    global roation
    roation = -0.4
    roation1 =0
    rotMatz =m.rotate(roation/2,m.vec3(0,0,1))
    rotMaty =m.rotate(roation/3,m.vec3(0,1,0))
    rotMatx =m.rotate(roation,m.vec3(1,0,0))

    tranx= m.translate(m.vec3(-(roation%3)/6,0,0))
    trany = m.translate(m.vec3(0,-2*(roation%3)/6,0))
    tranz =  m.translate(m.vec3(0,0,-2))

    pers = m.perspective(m.radians(50),r_scn,0.01,40)
    
    test = m.mat4([[-0.5,-0.5,0,1],[ 0.5, -0.5,0,1],[ 0.5, 0.5,0,1],[-0.5, 0.5,0,1]])
    # test=m.transpose(test)


    rotMat =  pers   * tranz 
    test2 = rotMat * m.vec4(0.5,0.5,0,1)
    test = rotMat * test
    # global roation
    # roation+=0.01
   
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

    
# Convert the resulting list of lists back to a PyGLM mat4 object
    result_matrix = m.mat3x4(matrix_list)
    
    loc =  glGetUniformLocation(program,'rot')
    # rotMat = m.mat4()
    glUniformMatrix4fv(loc,1,GL_FALSE,m.value_ptr(rotMat))
    
    
    
def load_image(image):
    
    # img = Image.open(image_path)
    img =  utils.convert_opencv_to_pil(image)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image in OpenGL context
    img_data = img.convert("RGBA").tobytes()    # Convert the image to bytes
    return img, img_data


def create_texture(image):
    
    img, img_data = load_image(image)
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    return texture_id


def mouse_button_callback(window, button, action, mods):
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        print("Left mouse button pressed" +str(glfw.get_cursor_pos(window)))
    elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        print("Right mouse button pressed")
    


# if __name__ == "__main__":
   
print('hello')
# roation=0
main()
    # extra()


