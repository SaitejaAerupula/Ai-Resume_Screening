// Day 1 - JavaScript Basics

console.log("Day 1 - JavaScript Practice");
console.log("---------------------------");

// Variables

var college = "ABC College";
let studentName = "Rahul";
const country = "India";

console.log("College:", college);
console.log("Student:", studentName);
console.log("Country:", country);

// Reassigning variables

college = "XYZ College";
studentName = "Rohit";

console.log("Updated College:", college);
console.log("Updated Student:", studentName);

// Data Types

let age = 22;
let isIntern = true;
let city;
let salary = null;
let skills = ["HTML", "CSS", "JavaScript"];
let person = {
    name: "Rahul",
    age: 22
};

console.log("\nData Types");
console.log("Number:", age);
console.log("String:", studentName);
console.log("Boolean:", isIntern);
console.log("Undefined:", city);
console.log("Null:", salary);
console.log("Array:", skills);
console.log("Object:", person);

// Arithmetic Operators

let a = 20;
let b = 5;

console.log("\nArithmetic Operators");
console.log("Addition:", a + b);
console.log("Subtraction:", a - b);
console.log("Multiplication:", a * b);
console.log("Division:", a / b);
console.log("Modulus:", a % b);

// Comparison Operators

console.log("\nComparison Operators");
console.log("a > b:", a > b);
console.log("a < b:", a < b);
console.log("a == b:", a == b);
console.log("a === b:", a === b);
console.log("a != b:", a != b);

// Logical Operators

let x = true;
let y = false;

console.log("\nLogical Operators");
console.log("x && y:", x && y);
console.log("x || y:", x || y);
console.log("!x:", !x);

// Template Literals

let language = "JavaScript";
let experience = "Beginner";

console.log("\nTemplate Literals");
console.log(`I am learning ${language}.`);
console.log(`Current Level: ${experience}`);
console.log(`Sum of ${a} and ${b} is ${a + b}.`);

// typeof Operator

console.log("\ntypeof Operator");
console.log(typeof age);
console.log(typeof studentName);
console.log(typeof isIntern);
console.log(typeof city);
console.log(typeof salary);
console.log(typeof skills);
console.log(typeof person);

// Simple Calculations

let length = 12;
let width = 8;

console.log("\nRectangle");
console.log("Length:", length);
console.log("Width:", width);
console.log("Area:", length * width);
console.log("Perimeter:", 2 * (length + width));

console.log("\nDay 1 Practice Completed");