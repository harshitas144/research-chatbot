## Objective
The project is meant to give you a refresher on aspects of React like props, functional components, state variables, and using dependencies. Don't feel pressured to make it perfect and remember to ask for help when you need it!

## Overview
You will be creating a basic React app that shows a MainPage component with a button displaying the text "Generate Post". When a user clicks the button, your app will make a fetch call to the JSONPlaceholder API to get a dummy user post.

Your code should render a new Post component under the button that displays the fetched data as text.
You can use any method you like for this, but we will have suggestions for each part of the project below.

## Instructions
****Please run `npm install` when starting the project. If you add any dependencies or extra libraries during the project please remember to also use  `npm install` to enable imports of these files. 

MainPage component

This will be the component displayed when the app loads. At first, there should only be a button with the text "Generate Post" visible on the page.

Use Material UI's pre-built button component https://mui.com/material-ui/react-button/ as your button component (make sure to npm install and import)

There should be a handleClick() function defined in MainPage that contains the logic for generating a new post.

If the user clicks the button multiple times, there should be MULTIPLE posts rendered sequentially (i.e. first click generates post 1, second click generates a different post below the first one).

## Suggestions
To test the project locally you can run `npm start` inside the react-project directory

To render multiple posts you can keep an array of post "id numbers" as a state variable, update it on each button click, and use forEach() to map through the array and render the Post components.
You can pass in a number as a prop to the Post component and use that in your API call inside the Post component to fetch a unique post.
Post component

You should make a fetch call to the JSONPlaceholder API to get the text for a post https://jsonplaceholder.typicode.com/guide/
A new post should be returned for each click (one way to do this is to increment a state variable in MainPage, pass it as a prop to Post, and use it in the API call).
You don't need much styling here, a simple container showing the post's text is fine!
Submission
To submit your learning sprint, complete the following steps:

Create components for Post and MainPage with the functionality described in the above sections.
Use git branch and make sure that it's your team name.
Add your changes by using `git add .`, commit your changes using `git commit -m "MESSAGE"`, and then push your changes using `git push --set-upstream main [YOUR_BRANCH]`.
And that's it, congratulations on finishing the React intro!.
