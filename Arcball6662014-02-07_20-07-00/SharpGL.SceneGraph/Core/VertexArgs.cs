using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpGL.SceneGraph.Core
{
    public class VertexArgs : EventArgs
    {
        public Vertex vertex;
        public int x;
        public int y;

        public VertexArgs(int x, int y, Vertex vertex)
        {
            this.x = x;
            this.y = y;
            this.vertex = vertex;
        }
    }
}
