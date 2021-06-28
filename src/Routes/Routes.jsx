import {BrowserRouter, Route} from 'react-router-dom'
import HomeComponent from '../Components/HomeComponent'

const Routes = () => {
    return(
        <BrowserRouter>
            <Route exact path='/' component={HomeComponent}></Route>
        </BrowserRouter>
    );
}

export default Routes;