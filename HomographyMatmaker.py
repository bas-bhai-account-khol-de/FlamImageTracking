import glfw
from OpenGL.GL import *
import numpy as np
import glm as m

roation  = 0


def main():
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
    position = glGetAttribLocation(shader_program, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(position)

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
    # roation = 15
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


    


# if __name__ == "__main__":
   
print('hello')
roation=0
main()
    # extra()


