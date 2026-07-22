const cart = [
  { id: 1, name: "Laptop", price: 50000, quantity: 1 },
  { id: 2, name: "Mouse", price: 800, quantity: 2 },
  { id: 3, name: "Keyboard", price: 1500, quantity: 1 },
  { id: 4, name: "Monitor", price: 12000, quantity: 1 }
];

//task 1
console.log("task 1: display all product names using map()");
cart.map(product => console.log(product.name));

//task 2
console.log("task 2 calculate the total price of all products in the cart using reduce()");
const totalPrice = cart.reduce((sum, product) => {
  return sum + (product.price * product.quantity);
}, 0);

//task 3
console.log("task 3 to find the product with id 3 using find()");
const product = cart.find(product => product.id === 3);
console.log(product);

//task 4
console.log("task 4 filter products greater than 5000 using filter()");
const expensiveProducts = cart.filter(product => product.price > 5000);
expensiveProducts.map(product => console.log(product.name));

//task 5.1
console.log("task 5.1 sort the products based on their price in ascending order using sort()");
const sortedProducts = cart.sort((a, b) => {
  return a.price - b.price;
});

//task 5.2
console.log("task 5.2 sort the products based on their quantity in descending order using sort()");
const sortedByQuantity = cart.sort((a, b) => {
  return b.quantity - a.quantity;
});

//task 6
console.log("task 6 using spread operator to create a new array with an additional product");
const newProduct = { id: 5, name: "Headphones", price: 2000, quantity: 1 };
const updatedCart = [...cart, newProduct];
console.log(updatedCart);

//task 7 
console.log("task 7 write a function using the rest operator to calculate the total price of any number of products");
function calculateTotal(...items) {
  return items.reduce((total, item) => {
    return total + (item.price * item.quantity);
  }, 0);
}

const total = calculateTotal(
  { name: "Laptop", price: 50000, quantity: 1 },
  { name: "Mouse", price: 800, quantity: 2 },
  { name: "Keyboard", price: 1500, quantity: 1 }
);

console.log("Total Price:",total);