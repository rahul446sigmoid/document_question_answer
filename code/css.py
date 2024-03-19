import streamlit as st

def render_navbar():
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
    st.markdown("""
    <head>
    <!-- Bootstrap CSS -->
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #92C5E8; height:80px; margin-top: 40px; margin-bottom: 20px;">
        <div class="container-fluid">
            <!-- Document AI -->
            <span class="navbar-text" style="color: black; margin-right: 20px;">Document AI</span>
            <!-- Add contact and about links on the right -->
            <div class="navbar-collapse justify-content-end">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="#" style="color: black;">Contact</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" style="color: black;">About</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    """, unsafe_allow_html=True)


